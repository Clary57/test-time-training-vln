"""CPU unit tests for episode-local trajectory-consensus TTRL.

Run with the project environment, for example:
    /nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python -m unittest \
        ttrl-src/tests/test_trajectory_consensus_updater.py
"""

import ast
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from unittest import mock
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

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                normalized = text.strip().lower()
                mapping = {
                    "forward 25 cm": [3],
                    "forward 50 cm": [4],
                    "forward 75 cm": [5],
                    "turn left 15 degree": [4],
                    "turn right 15 degree": [6],
                }
                return mapping.get(normalized, [7])

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
            self.generate_calls = 0
            self.generated_count = 0
            self.generate_configs = []

        def generate(self, input_ids, **kwargs):
            generation_config = kwargs["generation_config"]
            self.generate_configs.append(generation_config)
            count = generation_config.num_return_sequences
            indices = [
                (self.generated_count + offset) % len(self.completion_tokens)
                for offset in range(count)
            ]
            tokens = self.completion_tokens[indices]
            self.generated_count += count
            self.generate_calls += 1
            completion_ids = torch.stack(
                [tokens, torch.zeros(count, dtype=torch.long, device=input_ids.device)],
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

    def _updater(self, completion_tokens, completions, disable_update=False,
                 trace_metadata_enabled=False, sequential_generation=False, **kwargs):
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
            disable_update=disable_update,
            trace_metadata_enabled=trace_metadata_enabled,
            sequential_generation=sequential_generation,
            **kwargs,
        )
        return model, updater

    def test_sequential_generation_preserves_candidates_and_loss_update(self):
        model, updater = self._updater(
            [3, 3, 4],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>turn left 15 degree</answer>",
            },
            trace_metadata_enabled=True,
            sequential_generation=True,
        )
        updater.reset_episode("official-visual-input")
        before = model.lora_adapter.detach().clone()

        result = updater.adapt_step(self._inputs())

        self.assertEqual(model.generate_calls, 3)
        self.assertEqual(model.forward_use_cache, [False, False, False])
        self.assertEqual(result["candidate_actions"], [[1], [1], [2]])
        self.assertEqual(result["selected_actions"], [1])
        self.assertEqual(len(result["rewards"]), 3)
        self.assertFalse(torch.equal(model.lora_adapter.detach(), before))

    def test_default_sampling_remains_all_stochastic_in_one_batch(self):
        model, updater = self._updater(
            [3, 3, 4],
            {3: "forward 25 cm", 4: "turn left 15 degree"},
        )
        updater.reset_episode("default-sampling")
        result = updater.adapt_step(self._inputs())

        self.assertEqual(model.generate_calls, 1)
        self.assertEqual(model.generate_configs[0].num_return_sequences, 3)
        self.assertTrue(model.generate_configs[0].do_sample)
        self.assertEqual(result["candidate_sampling_modes"], ["stochastic"] * 3)

    def test_hybrid_sampling_is_greedy_then_stochastic_with_stable_order(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "forward 25 cm",
                4: "turn left 15 degree",
                5: "forward 75 cm",
            },
            candidate_sampling="greedy_plus_stochastic",
            trace_metadata_enabled=True,
        )
        updater.reset_episode("hybrid-sampling")
        result = updater.adapt_step(self._inputs())

        self.assertEqual(model.generate_calls, 3)
        self.assertFalse(model.generate_configs[0].do_sample)
        self.assertEqual(model.generate_configs[0].num_return_sequences, 1)
        self.assertTrue(model.generate_configs[1].do_sample)
        self.assertEqual(
            [config.num_return_sequences for config in model.generate_configs], [1, 1, 1]
        )
        self.assertEqual(result["candidate_actions"], [[1], [2], [1, 1, 1]])
        self.assertEqual(
            result["candidate_sampling_modes"], ["greedy", "stochastic", "stochastic"]
        )

    def test_hybrid_sequential_generation_keeps_one_config_per_candidate(self):
        model, updater = self._updater(
            [3, 4, 5],
            {3: "forward 25 cm", 4: "turn left 15 degree", 5: "forward 75 cm"},
            candidate_sampling="greedy_plus_stochastic",
            sequential_generation=True,
            trace_metadata_enabled=True,
        )
        updater.reset_episode("hybrid-sequential")
        result = updater.adapt_step(self._inputs())

        self.assertEqual([config.num_return_sequences for config in model.generate_configs], [1] * 3)
        self.assertEqual([config.do_sample for config in model.generate_configs], [False, True, True])
        self.assertEqual(result["candidate_actions"], [[1], [2], [1, 1, 1]])

    def test_se2_stop_distance_distinguishes_orientation_and_stop(self):
        _, legacy = self._updater([3], {3: "forward 25 cm"})
        _, se2 = self._updater(
            [3], {3: "forward 25 cm"}, reward_geometry="se2_stop"
        )
        left = legacy._trajectory([2])[-1]
        right = legacy._trajectory([3])[-1]
        self.assertEqual(legacy._trajectory_point_distance(left, right), 0.0)
        self.assertGreater(se2._trajectory_point_distance(
            se2._trajectory([2])[-1], se2._trajectory([3])[-1]
        ), 0.0)
        self.assertGreater(se2._trajectory_point_distance(
            se2._trajectory([0])[-1], se2._trajectory([])[-1]
        ), 0.0)

    def test_se2_reward_resolves_rotation_consensus_ignored_by_legacy_xy(self):
        _, legacy = self._updater([3], {3: "forward 25 cm"})
        _, se2 = self._updater(
            [3], {3: "forward 25 cm"}, reward_geometry="se2_stop"
        )
        actions = [[2], [2], [3]]
        self.assertEqual(legacy._consensus_rewards(actions, [True] * 3), [1.0] * 3)
        self.assertEqual(se2._consensus_rewards(actions, [True] * 3), [1.0, 1.0, -1.0])

    def test_hybrid_trace_on_off_preserves_candidates_actions_and_rng(self):
        outcomes = []
        for trace_enabled in (False, True):
            model, updater = self._updater(
                [3, 4, 5],
                {
                    3: "forward 25 cm",
                    4: "turn left 15 degree",
                    5: "forward 75 cm",
                },
                candidate_sampling="greedy_plus_stochastic",
                trace_metadata_enabled=trace_enabled,
            )
            updater.reset_episode("hybrid-trace-equivalence")
            result = updater.adapt_step(self._inputs())
            outcomes.append((
                result["selected_actions"], result["rewards"],
                [config.do_sample for config in model.generate_configs],
                random.getstate(), torch.random.get_rng_state(),
            ))
        self.assertEqual(outcomes[0][:4], outcomes[1][:4])
        self.assertTrue(torch.equal(outcomes[0][4], outcomes[1][4]))

    def test_se2_stop_trajectory_ends_at_first_stop(self):
        _, updater = self._updater(
            [3], {3: "forward 25 cm"}, reward_geometry="se2_stop"
        )
        trajectory = updater._trajectory([1, 0, 1, 2])
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(trajectory[-1][3], 1.0)

    def test_se2_angle_distance_wraps_at_pi(self):
        _, updater = self._updater(
            [3], {3: "forward 25 cm"}, reward_geometry="se2_stop"
        )
        first = [0.0, 0.0, math.radians(179), 0.0]
        second = [0.0, 0.0, math.radians(-179), 0.0]
        self.assertAlmostEqual(
            updater._trajectory_point_distance(first, second),
            updater.orientation_distance_scale * math.radians(2),
            places=7,
        )

    def test_new_controls_reject_invalid_values(self):
        for kwargs in (
            {"reward_geometry": "unknown"},
            {"candidate_sampling": "unknown"},
            {"orientation_distance_scale": -1.0},
            {"stop_distance_penalty": -1.0},
            {"cluster_quality_weight": 1.1},
            {"max_cluster_ratio": 1.1},
            {"min_cluster_quality_margin": -0.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._updater([3], {3: "forward 25 cm"}, **kwargs)

    def test_cluster_gate_accepts_two_vs_one_when_quality_is_disabled(self):
        model, updater = self._updater(
            [3, 3, 4],
            {3: "forward 25 cm", 4: "turn left 15 degree"},
            update_policy="cluster_gated",
            action_span_only=True,
            cluster_quality_weight=0.0,
            trace_metadata_enabled=True,
        )
        updater.reset_episode("cluster-only")
        result = updater.adapt_step(self._inputs())

        self.assertTrue(result["update_eligible"])
        self.assertTrue(result["optimizer_step_executed"])
        self.assertEqual(result["update_gate_reason"], "eligible_cluster")
        self.assertEqual(result["cluster_members"], [0, 1])
        self.assertEqual(result["cluster_outlier"], 2)
        self.assertEqual(result["policy_loss_type"], "cluster_vs_outlier_action_logp")
        self.assertEqual(result["selected_candidate_index"], 0)

    def test_cluster_quality_can_veto_a_low_quality_majority(self):
        _, updater = self._updater(
            [3, 3, 4],
            {3: "forward 25 cm", 4: "turn left 15 degree"},
            update_policy="cluster_gated",
            action_span_only=True,
            cluster_quality_weight=0.25,
            min_cluster_quality_margin=0.0,
        )
        updater.reset_episode("cluster-quality-veto")
        result = updater.adapt_step(self._inputs())

        self.assertFalse(result["update_eligible"])
        self.assertFalse(result["optimizer_step_executed"])
        self.assertEqual(result["update_gate_reason"], "cluster_quality_below_outlier")
        self.assertLess(result["cluster_quality_margin"], 0.0)

    def test_cluster_reference_scoring_does_not_change_default_behavior(self):
        outcomes = []
        for policy in ("legacy", "gated"):
            model, updater = self._updater(
                [3, 3, 4],
                {3: "forward 25 cm", 4: "turn left 15 degree"},
                update_policy=policy,
            )
            updater.reset_episode("default-noninterference")
            result = updater.adapt_step(self._inputs())
            outcomes.append((result["candidate_actions"] if "candidate_actions" in result else None,
                             result["selected_actions"], model.generate_calls))
        self.assertEqual(outcomes[0][1:], outcomes[1][1:])

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

    def test_disable_update_runs_consensus_and_backward_without_optimizer_step(self):
        model, updater = self._updater(
            [3, 3, 4],
            {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
            disable_update=True,
        )
        updater.reset_episode("frozen-update")
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        step_called = False
        original_step = updater.optimizer.step

        def tracked_step(*args, **kwargs):
            nonlocal step_called
            step_called = True
            return original_step(*args, **kwargs)

        updater.optimizer.step = tracked_step
        result = updater.adapt_step(self._inputs())

        self.assertFalse(step_called)
        self.assertFalse(result["update_enabled"])
        self.assertTrue(result["parameters_unchanged"])
        self.assertEqual(result["adapter_checksum_before"], result["adapter_checksum_after"])
        self.assertEqual(result["adapter_norm_before"], result["adapter_norm_after"])
        self.assertEqual(result["selected_actions"], [1])
        self.assertEqual(len(result["rewards"]), 3)
        self.assertIsInstance(result["loss"], float)
        self.assertTrue(model.lora_adapter.grad is not None)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), before[name]))

    def test_default_mode_still_updates_parameters(self):
        model, updater = self._updater(
            [3, 3, 4],
            {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
        )
        updater.reset_episode("default-update")
        before = model.lora_adapter.detach().clone()
        result = updater.adapt_step(self._inputs())
        self.assertTrue(result["update_enabled"])
        self.assertFalse(torch.equal(model.lora_adapter.detach(), before))

    def test_explicit_legacy_policy_matches_default_actions_loss_and_parameters(self):
        arguments = (
            [3, 3, 4],
            {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
        )
        results = []
        for kwargs in ({}, {"update_policy": "legacy"}):
            model, updater = self._updater(*arguments, **kwargs)
            updater.reset_episode("legacy-equivalence")
            result = updater.adapt_step(self._inputs())
            results.append((result, model.lora_adapter.detach().clone(), random.getstate()))
        self.assertEqual(results[0][0]["selected_actions"], results[1][0]["selected_actions"])
        self.assertEqual(results[0][0]["rewards"], results[1][0]["rewards"])
        self.assertEqual(results[0][0]["loss"], results[1][0]["loss"])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[0][2], results[1][2])

    def test_default_legacy_path_does_not_snapshot_or_compute_drift_guard(self):
        model, updater = self._updater(
            [3, 3, 4],
            {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
        )
        updater.reset_episode("legacy-no-protection-overhead")
        with mock.patch.object(
            updater, "_parameter_snapshot", side_effect=AssertionError("unexpected snapshot")
        ), mock.patch.object(
            updater, "_optimizer_snapshot", side_effect=AssertionError("unexpected snapshot")
        ), mock.patch.object(
            updater, "adapter_delta_norm", side_effect=AssertionError("unexpected drift check")
        ):
            result = updater.adapt_step(self._inputs())
        self.assertTrue(result["optimizer_step_executed"])
        self.assertIsNone(result["adapter_delta_norm"])
        self.assertFalse(result["update_rolled_back"])

    def test_gated_policy_skips_equal_rewards_and_tied_best(self):
        cases = (
            ([3, 3, 3], {3: "<answer>forward 25 cm</answer>"},
             "reward_std_below_threshold"),
            ([3, 3, 4], {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>turn left 15 degree</answer>",
            }, "best_candidate_not_unique"),
        )
        for tokens, completions, expected_reason in cases:
            model, updater = self._updater(tokens, completions, update_policy="gated")
            updater.reset_episode(expected_reason)
            before = model.lora_adapter.detach().clone()
            result = updater.adapt_step(self._inputs())
            self.assertFalse(result["update_eligible"])
            self.assertFalse(result["optimizer_step_executed"])
            self.assertFalse(result["update_enabled"])
            self.assertEqual(result["update_gate_reason"], expected_reason)
            self.assertTrue(torch.equal(model.lora_adapter.detach(), before))

    def test_gated_unique_medoid_updates_and_respects_episode_limit(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
            update_policy="gated",
            max_updates_per_episode=1,
        )
        updater.reset_episode("gated-limit")
        first = updater.adapt_step(self._inputs())
        second = updater.adapt_step(self._inputs())
        self.assertEqual(first["rewards"], [-1.0, 1.0, -1.0])
        self.assertTrue(first["update_eligible"])
        self.assertTrue(first["optimizer_step_executed"])
        self.assertTrue(first["effective_parameter_change"])
        self.assertNotEqual(first["adapter_checksum_before"], first["adapter_checksum_after"])
        self.assertEqual(first["policy_loss_type"], "best_vs_worst_action_logp")
        self.assertGreater(first["policy_loss"], 0.0)
        self.assertEqual(first["episode_optimizer_steps"], 1)
        self.assertFalse(second["update_eligible"])
        self.assertEqual(second["update_gate_reason"], "episode_update_limit_reached")
        self.assertFalse(second["optimizer_step_executed"])
        self.assertFalse(second["effective_parameter_change"])
        self.assertEqual(second["episode_optimizer_steps"], 1)

    def test_action_span_mask_selects_only_answer_tokens(self):
        _, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
            update_policy="gated",
            action_span_only=True,
        )
        ids = torch.tensor([9, 3, 10, 0], dtype=torch.long)
        base_mask = torch.ones(4)
        mask, available = updater._action_span_mask(
            ids, base_mask, "reason <answer>forward 25 cm</answer> trailing"
        )
        self.assertTrue(available)
        self.assertEqual(mask.tolist(), [0.0, 1.0, 0.0, 0.0])

        plain_mask, plain_available = updater._action_span_mask(
            torch.tensor([3, 4, 0, 0], dtype=torch.long),
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
            "forward 25 cm, forward 50 cm",
        )
        self.assertTrue(plain_available)
        self.assertEqual(plain_mask.tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_gated_action_span_unavailable_skips_without_fallback_training(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>unmapped forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
            update_policy="gated",
            action_span_only=True,
        )
        updater.reset_episode("missing-action-span")
        before = model.lora_adapter.detach().clone()
        result = updater.adapt_step(self._inputs())
        self.assertFalse(result["update_eligible"])
        self.assertEqual(result["update_gate_reason"], "action_span_unavailable")
        self.assertTrue(torch.equal(model.lora_adapter.detach(), before))

    def test_gated_drift_limit_rolls_back_rejected_step(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
            update_policy="gated",
            max_adapter_delta_norm=1e-12,
        )
        updater.reset_episode("rollback")
        before = model.lora_adapter.detach().clone()
        result = updater.adapt_step(self._inputs())
        self.assertTrue(result["update_eligible"])
        self.assertTrue(result["update_rolled_back"])
        self.assertEqual(result["rollback_reason"], "adapter_delta_norm_exceeded")
        self.assertFalse(result["optimizer_step_executed"])
        self.assertFalse(result["effective_parameter_change"])
        self.assertEqual(result["adapter_delta_norm"], 0.0)
        self.assertTrue(torch.equal(model.lora_adapter.detach(), before))
        self.assertFalse(updater.optimizer.state)

    def test_gated_anchor_penalty_and_rollback_preserve_prior_optimizer_state(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
            update_policy="gated",
            anchor_l2_weight=0.5,
        )
        updater.reset_episode("rollback-with-history")
        first = updater.adapt_step(self._inputs())
        self.assertTrue(first["optimizer_step_executed"])
        parameter_before_rejected_step = model.lora_adapter.detach().clone()
        optimizer_before = updater._optimizer_snapshot()
        updater.max_adapter_delta_norm = 1e-12

        second = updater.adapt_step(self._inputs())

        self.assertGreater(second["anchor_loss"], 0.0)
        self.assertTrue(second["update_rolled_back"])
        self.assertTrue(torch.equal(model.lora_adapter.detach(), parameter_before_rejected_step))
        optimizer_after = updater._optimizer_snapshot()
        self.assertEqual(optimizer_before["param_groups"], optimizer_after["param_groups"])
        self.assertEqual(optimizer_before["state"].keys(), optimizer_after["state"].keys())
        for key in optimizer_before["state"]:
            for field in optimizer_before["state"][key]:
                before = optimizer_before["state"][key][field]
                after = optimizer_after["state"][key][field]
                if torch.is_tensor(before):
                    self.assertTrue(torch.equal(before, after))
                else:
                    self.assertEqual(before, after)

    def test_gated_disable_update_keeps_parameters_but_reports_eligibility(self):
        model, updater = self._updater(
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
            disable_update=True,
            update_policy="gated",
        )
        updater.reset_episode("gated-disabled")
        before = model.lora_adapter.detach().clone()
        result = updater.adapt_step(self._inputs())
        self.assertTrue(result["update_eligible"])
        self.assertFalse(result["optimizer_step_executed"])
        self.assertFalse(result["update_enabled"])
        self.assertTrue(result["parameters_unchanged"])
        self.assertTrue(torch.equal(model.lora_adapter.detach(), before))

    def test_gated_trace_on_off_preserves_candidates_actions_parameters_and_rng(self):
        arguments = (
            [3, 4, 5],
            {
                3: "<answer>forward 25 cm</answer>",
                4: "<answer>forward 50 cm</answer>",
                5: "<answer>forward 75 cm</answer>",
            },
        )
        results = []
        for trace_enabled in (False, True):
            model, updater = self._updater(
                *arguments,
                trace_metadata_enabled=trace_enabled,
                update_policy="gated",
                action_span_only=True,
            )
            updater.reset_episode("gated-trace-noninterference")
            result = updater.adapt_step(self._inputs())
            results.append((result, model.lora_adapter.detach().clone(), random.getstate()))
        self.assertEqual(results[0][0]["selected_actions"], results[1][0]["selected_actions"])
        self.assertEqual(results[0][0]["rewards"], results[1][0]["rewards"])
        self.assertEqual(results[0][0]["loss"], results[1][0]["loss"])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[0][2], results[1][2])
        self.assertEqual(results[1][0]["selected_candidate_index"], 1)

    def test_trace_metadata_exposes_existing_medoid_without_changing_choice(self):
        arguments = (
            [3, 3, 4],
            {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
        )
        _, baseline = self._updater(*arguments)
        _, traced = self._updater(*arguments, trace_metadata_enabled=True)
        baseline.reset_episode("same")
        traced.reset_episode("same")
        baseline_result = baseline.adapt_step(self._inputs())
        traced_result = traced.adapt_step(self._inputs())
        self.assertEqual(baseline_result["selected_actions"], traced_result["selected_actions"])
        self.assertEqual(baseline_result["rewards"], traced_result["rewards"])
        self.assertNotIn("selected_candidate_index", baseline_result)
        self.assertEqual(traced_result["selected_candidate_index"], 0)
        self.assertEqual(traced_result["candidate_actions"], [[1], [1], [2]])

    def test_selected_candidate_index_is_computed_once_per_adapt_step(self):
        for enabled in (False, True):
            _, updater = self._updater(
                [3, 3, 4],
                {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
                trace_metadata_enabled=enabled,
            )
            updater.reset_episode(f"single-ranking-{enabled}")
            original = updater._selected_candidate_index
            calls = []

            def counted(rewards, valid):
                calls.append((list(rewards), list(valid)))
                return original(rewards, valid)

            updater._selected_candidate_index = counted
            result = updater.adapt_step(self._inputs())
            self.assertEqual(len(calls), 1)
            if enabled:
                self.assertEqual(result["selected_candidate_index"], 0)
                self.assertEqual(result["selected_actions"], result["candidate_actions"][0])
            else:
                self.assertNotIn("selected_candidate_index", result)

    def test_trace_on_off_valid_selection_and_rng_state_are_identical(self):
        arguments = (
            [3, 3, 4],
            {3: "<answer>forward 25 cm</answer>", 4: "<answer>turn left 15 degree</answer>"},
        )
        results = []
        for enabled in (False, True):
            _, updater = self._updater(*arguments, trace_metadata_enabled=enabled)
            updater.reset_episode("valid-rng")
            result = updater.adapt_step(self._inputs())
            results.append((result, random.getstate()))
        self.assertEqual(results[0][0]["selected_actions"], results[1][0]["selected_actions"])
        self.assertEqual(results[0][0]["rewards"], results[1][0]["rewards"])
        self.assertEqual(results[0][1], results[1][1])

    def test_trace_on_off_all_invalid_fallback_and_rng_state_are_identical(self):
        arguments = (
            [5, 5, 5],
            {5: "not a navigation action"},
        )
        results = []
        original_randint = random.randint
        for enabled in (False, True):
            _, updater = self._updater(*arguments, trace_metadata_enabled=enabled)
            updater.reset_episode("invalid-rng")
            with mock.patch.object(random, "randint", wraps=original_randint) as randint_mock:
                result = updater.adapt_step(self._inputs())
            results.append((result, random.getstate(), randint_mock.call_count))
        self.assertEqual(results[0][0]["selected_actions"], results[1][0]["selected_actions"])
        self.assertEqual(results[0][0]["rewards"], results[1][0]["rewards"])
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual(results[0][2], 1)
        self.assertEqual(results[1][2], 1)
        self.assertNotIn("selected_candidate_index", results[0][0])
        self.assertIsNone(results[1][0]["selected_candidate_index"])


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
        self.assertIn('parser.add_argument("--disable-update"', source)

    def test_frozen_evaluator_supports_ordered_episode_subset_without_changing_action_logic(self):
        source = (ROOT / "eval" / "eval.py").read_text()
        self.assertIn('parser.add_argument("--episode-ids-file"', source)
        self.assertIn("select_episodes_in_file_order", source)
        self.assertIn("shard_episodes", source)
        self.assertIn("dataset_split = dataset", source)
        self.assertIn('parser.add_argument("--max-executed-chunks"', source)
        self.assertIn("result[:self.max_executed_chunks]", source)
        self.assertNotIn("from vllm", source)
        self.assertIn("def encode_image_base64(image):", source)


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

    def test_explicit_expected_ids_merge_historical_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            old = temp / "old"
            new = temp / "new"
            self._write_shard(old, ["a"], {"a": {"success": 1}}, ["a"])
            self._write_shard(new, ["b"], {"b": {"success": 0}}, ["b"])
            result = AGGREGATOR.aggregate_result_dirs([old, new], ["a", "b"])
            self.assertEqual(result["episode_count"], 2)

    def test_explicit_expected_ids_reject_missing_unexpected_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            one = temp / "one"
            self._write_shard(one, ["a"], {"a": {"success": 1}}, ["a"])
            with self.assertRaisesRegex(ValueError, "missing"):
                AGGREGATOR.aggregate_result_dirs([one], ["a", "b"])
            with self.assertRaisesRegex(ValueError, "outside explicit"):
                AGGREGATOR.aggregate_result_dirs([one], ["b"])
            with self.assertRaisesRegex(ValueError, "unique"):
                AGGREGATOR.aggregate_result_dirs([one], ["a", "a"])

    def test_default_aggregation_still_rejects_different_manifest_expectations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            old = temp / "old"
            new = temp / "new"
            self._write_shard(old, ["a"], {"a": {"success": 1}}, ["a"])
            self._write_shard(new, ["b"], {"b": {"success": 0}}, ["b"])
            with self.assertRaisesRegex(ValueError, "disagree"):
                AGGREGATOR.aggregate_result_dirs([old, new])


if __name__ == "__main__":
    unittest.main()
