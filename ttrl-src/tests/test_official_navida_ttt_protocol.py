import ast
import unittest
from pathlib import Path


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

    def test_official_protocol_matches_public_double_image_input_construction(self):
        images, prompt_inputs = self._build_prompt_inputs(True)
        self.assertEqual(images, [["image"], ["image"]])
        self.assertEqual(prompt_inputs.device, "cpu")

    def test_official_protocol_is_explicit_and_requires_history_200(self):
        self.assertIn('parser.add_argument("--official-navida-protocol"', self.source)
        self.assertIn("args.official_navida_protocol and args.max_action_history != 200", self.source)
        self.assertIn('requires --max-action-history 200', self.source)

    def test_manifest_records_protocol_critical_parameters(self):
        for field in (
            "official_navida_protocol", "max_action_history", "max_executed_chunks",
            "sequential_candidate_generation",
            "num_generations", "temperature", "learning_rate", "weight_decay",
            "max_grad_norm", "disable_update", "stop_trace",
        ):
            self.assertIn(f'"{field}"', self.source)

    def test_consensus_updater_remains_limited_to_two_action_chunks(self):
        updater = EVALUATOR.parents[1] / "trainer" / "trajectory_consensus_updater.py"
        updater_source = updater.read_text()
        self.assertIn('completion.split(",") if chunk.strip()][:2]', updater_source)
        self.assertIn("sequential_generation=official_navida_protocol", self.source)


if __name__ == "__main__":
    unittest.main()
