"""CPU unit tests for episode-local trajectory-consensus TTRL.

Run with the project environment, for example:
    /nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python -m unittest \
        ttrl-src/tests/test_trajectory_consensus_updater.py
"""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - reported as a skipped suite on minimal hosts
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if torch is not None:
    from trainer.trajectory_consensus_updater import TrajectoryConsensusUpdater

AGGREGATOR_SPEC = importlib.util.spec_from_file_location(
    "aggregate_ttt_results", ROOT / "eval" / "aggregate_ttt_results.py"
)
AGGREGATOR = importlib.util.module_from_spec(AGGREGATOR_SPEC)
AGGREGATOR_SPEC.loader.exec_module(AGGREGATOR)


@unittest.skipIf(torch is None, "PyTorch is required for updater tests")
class TrajectoryConsensusUpdaterTests(unittest.TestCase):
    class FakeProcessor:
        class Tokenizer:
            eos_token_id = 0

        def __init__(self, completions):
            self.tokenizer = self.Tokenizer()
            self.completions = completions

        def batch_decode(self, completion_ids, skip_special_tokens=True):
            return [self.completions[row[0].item()] for row in completion_ids]

    class FakeModel(torch.nn.Module):
        def __init__(self, completion_tokens):
            super().__init__()
            self.lora_adapter = torch.nn.Parameter(torch.tensor(0.25))
            self.visual_frozen = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)
            self.register_buffer("completion_tokens", torch.tensor(completion_tokens, dtype=torch.long))
            self.register_buffer("class_offsets", torch.linspace(-1.0, 1.0, 8))
            self.forward_use_cache = []

        def generate(self, input_ids, **kwargs):
            count = self.completion_tokens.numel()
            completion_ids = torch.stack(
                [self.completion_tokens, torch.zeros(count, dtype=torch.long, device=input_ids.device)],
                dim=1,
            )
            return torch.cat([input_ids.repeat(count, 1), completion_ids], dim=1)

        def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, use_cache):
            self.forward_use_cache.append(use_cache)
            batch_size, sequence_length = input_ids.shape
            logits = (self.lora_adapter * self.class_offsets).view(1, 1, -1)
            logits = logits.expand(batch_size, sequence_length, -1)
            return SimpleNamespace(logits=logits)

    @staticmethod
    def _inputs():
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": torch.zeros((1, 1), dtype=torch.float32),
            "image_grid_thw": torch.zeros((1, 3), dtype=torch.long),
        }

    def _updater(self, completion_tokens, completions):
        model = self.FakeModel(completion_tokens)
        updater = TrajectoryConsensusUpdater(
            model,
            self.FakeProcessor(completions),
            learning_rate=0.1,
            weight_decay=0.0,
            max_grad_norm=1.0,
            num_generations=len(completion_tokens),
            temperature=0.3,
            seed=123,
        )
        return model, updater

    def test_adapt_step_updates_only_lora_and_reset_restores_base_and_optimizer(self):
        model, updater = self._updater(
            [3, 3, 4],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>turn left 15 degree</answer>",
            },
        )
        updater.reset_episode("episode-a")
        base_adapter = model.lora_adapter.detach().clone()
        base_visual = model.visual_frozen.detach().clone()
        initial_optimizer = updater.optimizer

        result = updater.adapt_step(self._inputs())

        self.assertEqual(result["valid_count"], 3)
        self.assertEqual(result["selected_actions"], [1])
        self.assertFalse(torch.equal(model.lora_adapter.detach(), base_adapter))
        self.assertTrue(torch.equal(model.visual_frozen.detach(), base_visual))
        self.assertTrue(updater.optimizer.state)
        self.assertEqual(model.forward_use_cache, [False])

        updater.reset_episode("episode-b")
        self.assertIsNot(updater.optimizer, initial_optimizer)
        self.assertFalse(updater.optimizer.state)
        self.assertTrue(torch.equal(model.lora_adapter.detach(), base_adapter))
        self.assertTrue(torch.equal(model.visual_frozen.detach(), base_visual))

    def test_second_adapt_step_uses_the_first_step_updated_adapter(self):
        model, updater = self._updater(
            [3, 3, 4],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>turn left 15 degree</answer>",
            },
        )
        updater.reset_episode("same-episode")
        first_result = updater.adapt_step(self._inputs())
        second_result = updater.adapt_step(self._inputs())

        self.assertNotEqual(first_result["adapter_norm_before"], first_result["adapter_norm_after"])
        self.assertAlmostEqual(
            second_result["adapter_norm_before"], first_result["adapter_norm_after"], places=6
        )

    def test_equal_rewards_produce_no_effective_update(self):
        model, updater = self._updater(
            [3, 3, 3],
            {3: "<answer>forward 25 cm</answer>"},
        )
        updater.reset_episode("equal-reward")
        before = model.lora_adapter.detach().clone()

        result = updater.adapt_step(self._inputs())

        self.assertEqual(result["rewards"], [1.0, 1.0, 1.0])
        self.assertTrue(torch.equal(model.lora_adapter.detach(), before))

    def test_invalid_candidates_do_not_break_consensus_or_contribute_gradients(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>turn left 15 degree</answer>",
                5: "not a navigation action",
            },
        )
        updater.reset_episode("invalid-candidate")

        result = updater.adapt_step(self._inputs())

        self.assertEqual(result["valid_count"], 2)
        self.assertEqual(result["rewards"][2], -1.0)
        self.assertIn(result["selected_actions"], ([1], [2]))


class EvaluatorProtocolSourceTests(unittest.TestCase):
    def test_evaluator_normalizes_episode_ids_before_sharding_manifest_and_environment(self):
        source = (ROOT / "eval" / "eval_ttt.py").read_text()
        tree = ast.parse(source)
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        normalize_node = functions["normalize_episode_ids"]
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[normalize_node], type_ignores=[])),
                filename="eval_ttt.py",
                mode="exec",
            ),
            namespace,
        )
        episodes = [
            SimpleNamespace(episode_id=1, marker=object()),
            SimpleNamespace(episode_id="2", marker=object()),
        ]
        original_episodes = list(episodes)
        original_markers = [episode.marker for episode in episodes]
        dataset = SimpleNamespace(episodes=episodes)

        namespace["normalize_episode_ids"](dataset)

        self.assertEqual([episode.episode_id for episode in dataset.episodes], ["1", "2"])
        self.assertEqual(dataset.episodes, original_episodes)
        self.assertEqual([episode.marker for episode in dataset.episodes], original_markers)

        main_node = functions["main"]
        normalize_call = next(
            node for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "normalize_episode_ids"
        )
        all_episodes_assignment = next(
            node for node in ast.walk(main_node)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "all_episodes" for target in node.targets)
        )
        manifest_call = next(
            node for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_episode_manifest"
        )
        evaluate_call = next(
            node for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "evaluate_agent"
        )
        self.assertLess(normalize_call.lineno, all_episodes_assignment.lineno)
        self.assertLess(normalize_call.lineno, manifest_call.lineno)
        self.assertLess(normalize_call.lineno, evaluate_call.lineno)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Env"
                for node in ast.walk(functions["evaluate_agent"])
            )
        )

    def test_pending_action_fast_path_precedes_online_update(self):
        source = (ROOT / "eval" / "eval_ttt.py").read_text()
        act_start = source.index("    def act(")
        act_end = source.index("\n\ndef main", act_start)
        act_source = source[act_start:act_end]
        self.assertLess(
            act_source.index("if len(self.pending_action_list) != 0"),
            act_source.index("self.updater.adapt_step"),
        )

    def test_evaluator_preserves_trainable_lora_and_writes_manifest(self):
        source = (ROOT / "eval" / "eval_ttt.py").read_text()
        self.assertNotIn("merge_and_unload", source)
        self.assertNotIn("from vllm", source)
        self.assertIn("def encode_image_base64(image):", source)
        self.assertIn("write_episode_manifest(", source)
        self.assertIn("TrajectoryConsensusUpdater", source)
        self.assertIn('parser.add_argument("--max-episodes"', source)
        self.assertIn("expected_episode_ids", source)
        self.assertIn("all_episodes[shard_id::args.split_num]", source)
        self.assertIn("early_stop_steps=400", source)


class ResultAggregationTests(unittest.TestCase):
    @staticmethod
    def _write_shard(root, episode_ids, records, expected_episode_ids=None):
        root.mkdir()
        (root / "log").mkdir()
        manifest = {"episode_ids": episode_ids}
        if expected_episode_ids is not None:
            manifest["expected_episode_ids"] = expected_episode_ids
        (root / "episode_manifest.json").write_text(json.dumps(manifest))
        for episode_id, record in records.items():
            payload = dict(record, id=episode_id)
            (root / "log" / f"stats_{episode_id}.json").write_text(json.dumps(payload))

    def test_aggregate_recomputes_metric_means_from_raw_episode_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            shard_zero = temp / "split_0"
            shard_one = temp / "split_1"
            self._write_shard(
                shard_zero,
                ["a", "c"],
                {
                    "a": {"success": 1, "spl": 0.5, "distance_to_goal": 1.0},
                    "c": {"success": 0, "spl": 0.0, "distance_to_goal": 5.0},
                },
            )
            self._write_shard(
                shard_one,
                ["b"],
                {"b": {"success": 1, "spl": 1.0, "distance_to_goal": 2.0}},
            )

            summary = AGGREGATOR.aggregate_result_dirs([shard_zero, shard_one])

            self.assertEqual(summary["episode_count"], 3)
            self.assertEqual(summary["episode_ids"], ["a", "b", "c"])
            self.assertAlmostEqual(summary["metrics"]["success"]["mean"], 2.0 / 3.0)
            self.assertAlmostEqual(summary["metrics"]["spl"]["mean"], 0.5)
            self.assertAlmostEqual(summary["metrics"]["distance_to_goal"]["mean"], 8.0 / 3.0)

    def test_aggregate_rejects_an_entire_missing_shard_from_global_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            split_zero = temp / "split_0"
            self._write_shard(
                split_zero,
                ["a"],
                {"a": {"success": 1}},
                expected_episode_ids=["a", "b"],
            )
            with self.assertRaisesRegex(ValueError, "missing=.*b"):
                AGGREGATOR.aggregate_result_dirs([split_zero])

    def test_aggregate_rejects_missing_and_duplicate_episode_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            missing = temp / "missing"
            self._write_shard(missing, ["a", "b"], {"a": {"success": 1}})
            with self.assertRaisesRegex(ValueError, "missing"):
                AGGREGATOR.aggregate_result_dirs([missing])

            first = temp / "first"
            second = temp / "second"
            self._write_shard(first, ["x"], {"x": {"success": 1}})
            self._write_shard(second, ["x"], {"x": {"success": 0}})
            with self.assertRaisesRegex(ValueError, "multiple manifests"):
                AGGREGATOR.aggregate_result_dirs([first, second])


if __name__ == "__main__":
    unittest.main()
