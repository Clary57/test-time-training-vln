import ast
import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


TTRL_SRC = Path(__file__).resolve().parents[1]
EVAL_DIR = TTRL_SRC / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from episode_subset import (  # noqa: E402
    load_episode_ids_file,
    select_episodes_in_file_order,
    shard_episodes,
)


EVALUATOR = EVAL_DIR / "eval_official_subset.py"
OFFICIAL_COMMIT = "b86316a78a2865c80b445ed5a8fcc08fc0243dfb"
OFFICIAL_SHA256 = "790ecf3911acaf35b422ef93c18c7a7ffacd55712f5292a60b5cbb53790ca39a"


class Episode:
    def __init__(self, episode_id):
        self.episode_id = episode_id


class OfficialNaVIDASubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EVALUATOR.read_text()
        cls.tree = ast.parse(cls.source)

    def test_source_provenance_is_locked(self):
        self.assertIn(f'OFFICIAL_NAVIDA_COMMIT = "{OFFICIAL_COMMIT}"', self.source)
        self.assertIn(f'OFFICIAL_EVALUATOR_SHA256 = "{OFFICIAL_SHA256}"', self.source)

    def test_official_action_and_history_semantics_are_preserved(self):
        self.assertIn("select_action_idx = 2", self.source)
        self.assertIn("result = result[:select_action_idx]", self.source)
        self.assertIn('parser.add_argument("--max-action-history"', self.source)
        self.assertIn("default=10", self.source)
        self.assertNotIn("max-executed-chunks", self.source)

    def test_subset_cli_is_the_only_new_evaluator_control(self):
        self.assertIn('parser.add_argument("--episode-ids-file"', self.source)
        self.assertIn("if args.episode_ids_file is None:", self.source)
        self.assertIn(
            "dataset.get_splits(args.split_num,allow_uneven_splits=True)[args.split_id]",
            self.source,
        )

    def test_vllm_compatibility_helper_preserves_jpeg_base64_contract(self):
        function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "encode_image_base64"
        )
        namespace = {"io": io, "base64": base64}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(EVALUATOR), "exec"), namespace)

        class FakeImage:
            def save(self, buffer, format):
                self.format = format
                buffer.write(b"jpeg-fixture")

        image = FakeImage()
        encoded = namespace["encode_image_base64"](image)
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(encoded, base64.b64encode(b"jpeg-fixture").decode("utf-8"))

    def test_selection_preserves_requested_order_and_strided_shards(self):
        episodes = [Episode("3"), Episode("1"), Episode("4"), Episode("2")]
        selected = select_episodes_in_file_order(episodes, ["1", "2", "3", "4"])
        self.assertEqual([e.episode_id for e in selected], ["1", "2", "3", "4"])
        self.assertEqual([e.episode_id for e in shard_episodes(selected, 2, 0)], ["1", "3"])
        self.assertEqual([e.episode_id for e in shard_episodes(selected, 2, 1)], ["2", "4"])

    def test_selection_hash_is_raw_file_sha256(self):
        raw = b'{"episode_ids":["694","1428"]}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_bytes(raw)
            ids, metadata, digest = load_episode_ids_file(path)
        self.assertEqual(ids, ["694", "1428"])
        self.assertEqual(metadata, {})
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_manifest_records_official_identity_and_subset(self):
        function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_subset_manifest"
        )
        text = ast.get_source_segment(self.source, function)
        for field in (
            "official_repository", "official_commit", "official_evaluator_sha256",
            "episode_ids_file_sha256", "expected_episode_ids", "episode_ids",
            "max_action_history", "action_chunks",
        ):
            self.assertIn(f'"{field}"', text)

    def test_invalid_or_duplicate_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps({"episode_ids": ["1", "1"]}))
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_episode_ids_file(path)


if __name__ == "__main__":
    unittest.main()
