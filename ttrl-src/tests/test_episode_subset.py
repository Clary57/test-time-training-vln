import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("episode_subset", ROOT / "eval" / "episode_subset.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EpisodeSubsetTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "selection.json"
            path.write_text(json.dumps(payload))
            return MODULE.load_episode_ids_file(path)

    def test_order_and_integer_string_normalization(self):
        ids, metadata, digest = self._load({"episode_ids": [3, "1", 2], "metadata": {"seed": 41}})
        self.assertEqual(ids, ["3", "1", "2"])
        self.assertEqual(metadata["seed"], 41)
        self.assertEqual(len(digest), 64)
        episodes = [SimpleNamespace(episode_id=str(i)) for i in range(1, 4)]
        self.assertEqual([x.episode_id for x in MODULE.select_episodes_in_file_order(episodes, ids)], ids)

    def test_duplicate_unknown_and_empty_are_rejected(self):
        for payload, message in [([1, "1"], "duplicates"), ([], "empty")]:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                self._load(payload)
        with self.assertRaisesRegex(ValueError, "not found"):
            MODULE.select_episodes_in_file_order([SimpleNamespace(episode_id="1")], ["2"])

    def test_invalid_id_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strings or integers"):
            self._load([True])

    def test_twenty_ids_twenty_shards_cover_exactly_once(self):
        episodes = [SimpleNamespace(episode_id=str(i)) for i in range(20)]
        shards = [MODULE.shard_episodes(episodes, 20, split_id) for split_id in range(20)]
        self.assertTrue(all(len(shard) == 1 for shard in shards))
        self.assertEqual({x.episode_id for shard in shards for x in shard}, {str(i) for i in range(20)})

    def test_selection_smaller_than_split_count_would_have_empty_shards(self):
        episodes = [SimpleNamespace(episode_id=str(i)) for i in range(2)]
        shards = [MODULE.shard_episodes(episodes, 3, split_id) for split_id in range(3)]
        self.assertEqual([len(shard) for shard in shards], [1, 1, 0])

    def test_legacy_modulo_prefix_behavior_is_unchanged(self):
        episodes = list(range(10))
        old = episodes[1::3][:2]
        self.assertEqual(MODULE.shard_episodes(episodes, 3, 1)[:2], old)

    def test_evaluator_rejects_episode_file_with_max_episodes(self):
        source = (ROOT / "eval" / "eval_ttt.py").read_text()
        self.assertIn("args.episode_ids_file is not None and args.max_episodes is not None", source)
        self.assertIn("cannot be used together", source)


if __name__ == "__main__":
    unittest.main()
