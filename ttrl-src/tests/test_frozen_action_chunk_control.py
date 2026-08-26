import importlib.util
import inspect
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
EVALUATOR = ROOT / "eval" / "eval.py"
sys.path.insert(0, str(EVALUATOR.parent))
SPEC = importlib.util.spec_from_file_location("frozen_eval_factor_control", EVALUATOR)
frozen_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen_eval)


class FrozenActionChunkControlTests(unittest.TestCase):
    @staticmethod
    def agent(max_executed_chunks):
        agent = frozen_eval.NaVIDA_Agent.__new__(frozen_eval.NaVIDA_Agent)
        agent.max_executed_chunks = frozen_eval.normalize_max_executed_chunks(
            max_executed_chunks
        )
        agent.forward_distance = 25
        agent.turn_angle = 15
        agent.pending_action_list = []
        return agent

    def test_default_and_explicit_three_are_equivalent(self):
        self.assertEqual(
            inspect.signature(frozen_eval.NaVIDA_Agent.__init__).parameters[
                "max_executed_chunks"
            ].default,
            3,
        )
        parsed = [(1, 50), (2, 30), (3, 15), (1, 25)]
        default = self.agent(3)._queue_action_chunks(parsed, "raw")
        explicit = self.agent(3)._queue_action_chunks(parsed, "raw")
        self.assertEqual(default, explicit)
        self.assertEqual(default, [1, 1, 2, 2, 3])

    def test_two_only_removes_third_chunk(self):
        parsed = [(1, 50), (2, 30), (3, 15)]
        two = self.agent(2)._queue_action_chunks(parsed, "raw")
        three = self.agent(3)._queue_action_chunks(parsed, "raw")
        self.assertEqual(two, [1, 1, 2, 2])
        self.assertEqual(three[: len(two)], two)
        self.assertEqual(three[len(two) :], [3])

    def test_stop_keeps_original_pending_order(self):
        parsed = [(1, 25), (0, None), (3, 15)]
        self.assertEqual(self.agent(2)._queue_action_chunks(parsed, "raw"), [1, 0])
        self.assertEqual(self.agent(3)._queue_action_chunks(parsed, "raw"), [1, 0, 3])

    def test_invalid_fallback_rng_call_count_and_state(self):
        parsed = [(None, None), (1, 25), (None, None)]
        calls = []
        original = frozen_eval.random.randint

        def randint(low, high):
            calls.append((low, high))
            return original(low, high)

        frozen_eval.random.randint = randint
        try:
            random.seed(73)
            two = self.agent(2)._queue_action_chunks(parsed, "raw")
            two_state = random.getstate()
            two_calls = list(calls)
            calls.clear()
            random.seed(73)
            three = self.agent(3)._queue_action_chunks(parsed, "raw")
            three_state = random.getstate()
            three_calls = list(calls)
        finally:
            frozen_eval.random.randint = original
        self.assertEqual(len(two_calls), 1)
        self.assertEqual(len(three_calls), 2)
        self.assertEqual(two, three[: len(two)])
        self.assertNotEqual(two_state, three_state)

    def test_trace_metadata_is_rng_and_action_neutral(self):
        parsed = [(None, None), (1, 25), (0, None)]
        random.seed(91)
        actions_off = self.agent(3)._queue_action_chunks(parsed, "raw")
        state_off = random.getstate()
        random.seed(91)
        actions_on = self.agent(3)._queue_action_chunks(parsed, "raw")
        rows = frozen_eval.chunk_metadata(
            episode_id="x",
            decision_id=0,
            selected_chunk=actions_on,
            raw_model_outputs=["invalid, forward, stop"],
            parsed_candidate_chunks=[actions_on],
            candidate_rewards=[],
            selected_index=0,
        )
        for row in rows:
            row["max_executed_chunks"] = 3
        self.assertEqual(actions_off, actions_on)
        self.assertEqual(state_off, random.getstate())
        self.assertTrue(all(row["max_executed_chunks"] == 3 for row in rows))

    def test_manifest_and_run_metadata_record_effective_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode = SimpleNamespace(episode_id="7")
            frozen_eval.write_episode_manifest(
                root, 1, 0, [episode], ["7"], "hash", 41, "val_unseen", 2
            )
            frozen_eval.record_run_metadata(root, 2)
            self.assertEqual(
                json.loads((root / "episode_manifest.json").read_text())[
                    "max_executed_chunks"
                ],
                2,
            )
            self.assertEqual(
                json.loads((root / "run_metadata.json").read_text())[
                    "max_executed_chunks"
                ],
                2,
            )

    def test_run_metadata_preserves_existing_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "run_metadata.json").write_text(json.dumps({"episode_id": "7"}))
            frozen_eval.record_run_metadata(root, 3)
            self.assertEqual(
                json.loads((root / "run_metadata.json").read_text()),
                {"episode_id": "7", "max_executed_chunks": 3},
            )

    def test_cli_help_and_rejects_invalid_value(self):
        help_result = subprocess.run(
            [sys.executable, str(EVALUATOR), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--max-executed-chunks {2,3}", help_result.stdout)
        invalid = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--exp-config", "unused",
                "--split-num", "1",
                "--split-id", "0",
                "--model-path", "unused",
                "--result-path", "unused",
                "--max-executed-chunks", "4",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid choice: 4", invalid.stderr)

    def test_source_default_and_early_stop_are_locked(self):
        source = EVALUATOR.read_text()
        self.assertIn('choices=(2, 3),default=3', source)
        self.assertIn("EARLY_STOP_ROTATION = 25", source)
        self.assertIn("EARLY_STOP_STEPS = 400", source)
        self.assertIn('metadata["max_executed_chunks"]', source)
        self.assertNotIn("select_action_idx = 3", source)


if __name__ == "__main__":
    unittest.main()
