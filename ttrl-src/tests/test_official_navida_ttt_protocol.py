import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


EVALUATOR = Path(__file__).resolve().parents[1] / "eval" / "eval_ttt.py"


class FakePromptInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.images = None

    def apply_chat_template(self, conversations, **kwargs):
        return "prompt"

    def __call__(self, *, text, images, return_tensors, padding):
        self.images = images
        return FakePromptInputs()


class OfficialNaVIDATTRLProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EVALUATOR.read_text()
        cls.tree = ast.parse(cls.source)

    def _build_prompt_inputs(self, official):
        agent = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NaVIDA_Agent"
        )
        method = next(
            node for node in agent.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_prompt_inputs"
        )
        namespace = {"process_vision_info": lambda conversations: (["image"], [])}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(EVALUATOR), "exec"), namespace)

        instance = type("Agent", (), {})()
        instance.processor = FakeProcessor()
        instance.conversations = ["conversation"]
        instance.device = "cpu"
        instance.official_navida_protocol = official
        result = namespace["build_prompt_inputs"](instance)
        return instance.processor.images, result

    def test_legacy_prompt_inputs_remain_unchanged_by_default(self):
        images, prompt_inputs = self._build_prompt_inputs(False)
        self.assertEqual(images, [["image"]])
        self.assertEqual(prompt_inputs.device, "cpu")

    def test_official_protocol_matches_public_single_image_input_construction(self):
        images, prompt_inputs = self._build_prompt_inputs(True)
        self.assertEqual(images, [["image"]])
        self.assertEqual(prompt_inputs.device, "cpu")

    def test_official_protocol_is_explicit_and_requires_history_200(self):
        self.assertIn('parser.add_argument("--official-navida-protocol"', self.source)
        self.assertIn("args.official_navida_protocol and args.max_action_history != 200", self.source)
        self.assertIn('requires --max-action-history 200', self.source)

    def test_manifest_records_protocol_critical_parameters(self):
        for field in (
            "official_navida_protocol", "ttrl_profile", "max_action_history", "max_executed_chunks",
            "sequential_candidate_generation",
            "num_generations", "temperature", "learning_rate", "weight_decay",
            "max_grad_norm", "disable_update", "stop_trace",
            "update_policy", "min_reward_std", "min_reward_margin",
            "action_span_only", "max_updates_per_episode", "anchor_l2_weight",
            "max_adapter_delta_norm", "pairwise_beta",
            "reward_geometry", "orientation_distance_scale",
            "stop_distance_penalty", "candidate_sampling",
            "cluster_quality_weight", "max_cluster_ratio",
            "min_cluster_quality_margin",
        ):
            self.assertIn(f'"{field}"', self.source)

    def test_named_a5_profile_is_a4_plus_geometry_and_sampling_only(self):
        profile_assignment = next(
            node for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "TTRL_PROFILES"
                    for target in node.targets)
        )
        profile_function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "apply_ttrl_profile"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[profile_assignment, profile_function], type_ignores=[]),
                str(EVALUATOR), "exec",
            ),
            namespace,
        )
        profiles = namespace["TTRL_PROFILES"]
        differing_fields = {
            name for name in profiles["a4"]
            if profiles["a4"][name] != profiles["a5"][name]
        }
        self.assertEqual(differing_fields, {"reward_geometry", "candidate_sampling"})
        self.assertEqual(profiles["a5"]["reward_geometry"], "se2_stop")
        self.assertEqual(profiles["a5"]["candidate_sampling"], "greedy_plus_stochastic")

        custom = SimpleNamespace(ttrl_profile="custom", update_policy="legacy")
        self.assertIs(namespace["apply_ttrl_profile"](custom), custom)
        self.assertEqual(custom.update_policy, "legacy")
        a5 = SimpleNamespace(ttrl_profile="a5")
        namespace["apply_ttrl_profile"](a5)
        for name, value in profiles["a5"].items():
            self.assertEqual(getattr(a5, name), value)

    def test_gated_update_controls_are_explicit_and_legacy_is_default(self):
        self.assertIn(
            'parser.add_argument("--update-policy",choices=("legacy", "gated", "cluster_gated"),default="legacy"',
            self.source,
        )
        for option in (
            "--min-reward-std", "--min-reward-margin", "--action-span-only",
            "--max-updates-per-episode", "--anchor-l2-weight",
            "--max-adapter-delta-norm", "--pairwise-beta",
            "--reward-geometry", "--orientation-distance-scale",
            "--stop-distance-penalty", "--candidate-sampling",
            "--cluster-quality-weight", "--max-cluster-ratio",
            "--min-cluster-quality-margin",
        ):
            self.assertIn(option, self.source)

    def test_consensus_updater_remains_limited_to_two_action_chunks(self):
        updater = EVALUATOR.parents[1] / "trainer" / "trajectory_consensus_updater.py"
        updater_source = updater.read_text()
        self.assertIn('completion.split(",") if chunk.strip()][:2]', updater_source)
        self.assertIn("sequential_generation=official_navida_protocol", self.source)

    def test_native_adapter_keeps_qwen_type_and_is_initially_output_neutral(self):
        config = Qwen2_5_VLConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=62,
            vision_end_token_id=63,
            rope_scaling={"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
            vision_config={
                "depth": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "out_hidden_size": 16,
                "fullatt_block_indexes": [0], "patch_size": 2,
                "spatial_merge_size": 1, "temporal_patch_size": 1,
                "in_channels": 3, "window_size": 8,
            },
        )
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        original_type = type(model)
        inputs = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        with torch.no_grad():
            before = model(**inputs).logits.clone()
        model.add_adapter(LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            exclude_modules=".*visual.*",
        ))
        with torch.no_grad():
            after = model(**inputs).logits
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertIs(type(model), original_type)
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))
        self.assertFalse(any("visual" in name for name in trainable))
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
