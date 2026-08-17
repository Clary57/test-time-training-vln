import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("selector", SCRIPTS / "select_representative_r2r_episodes.py")
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


class BalancedExtensionTests(unittest.TestCase):
    def test_capacity_proportional_allocation_is_exact_and_deterministic(self):
        capacities = {"large": 80, "medium": 15, "small": 5}
        first = SELECTOR.allocate_capacity_proportional(capacities, 20, 41, "short")
        second = SELECTOR.allocate_capacity_proportional(capacities, 20, 41, "short")
        self.assertEqual(first, second)
        self.assertEqual(sum(first[0].values()), 20)
        self.assertEqual(first[0], {"large": 16, "medium": 3, "small": 1})

    def test_extension_excludes_old_ids_and_hits_class_targets(self):
        records = []
        for scene in ("a", "b"):
            for index in range(30):
                records.append({"episode_id": f"{scene}{index:02d}", "scene_id": scene,
                                "path_length": index, "instruction_length": 1})
        old = ["a00", "a10", "a20"]
        selected, audit = SELECTOR.select_balanced_extension(
            records, old, {"short": 4, "middle": 3, "long": 4}, 41
        )
        self.assertFalse(set(old) & {item["episode_id"] for item in selected})
        self.assertEqual(Counter(item["path_length_bin"] for item in selected),
                         {"short": 4, "middle": 3, "long": 4})
        self.assertEqual(sum(section["selected"] for section in audit.values()), 11)


if __name__ == "__main__":
    unittest.main()
