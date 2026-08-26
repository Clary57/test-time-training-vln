#!/usr/bin/env python3
import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EXPECTED_SELECTION_SHA = "e3d3c050d471ca86a253ca8238cd662a6208e72c10ed0bdb2248390329becfb6"
EXPECTED_OFFICIAL_SHA = "790ecf3911acaf35b422ef93c18c7a7ffacd55712f5292a60b5cbb53790ca39a"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection_path = ROOT / "selection128.json"
    selection = json.loads(selection_path.read_text())
    ids = [str(value) for value in selection["episode_ids"]]
    assert sha256(selection_path) == EXPECTED_SELECTION_SHA
    assert len(ids) == 128 and len(set(ids)) == 128

    evaluator = PROJECT / "ttrl-src/eval/eval_official_subset.py"
    source = evaluator.read_text()
    ast.parse(source)
    assert f'OFFICIAL_EVALUATOR_SHA256 = "{EXPECTED_OFFICIAL_SHA}"' in source
    assert "select_action_idx = 2" in source
    assert 'parser.add_argument("--episode-ids-file"' in source
    assert "max-executed-chunks" not in source

    launch = ROOT / "launch.sh"
    subprocess.run(["bash", "-n", str(launch)], check=True)
    launch_text = launch.read_text()
    for required in (
        "SPLIT_NUM=8", "--max-action-history 200", "--num-generations 1",
        "--episode-ids-file", "GPU $gpu has compute processes",
        "GPU $gpu has only ${free_mib} MiB free",
    ):
        assert required in launch_text
    assert "--lora-path" not in launch_text
    print(json.dumps({
        "status": "PASS",
        "selection_sha256": sha256(selection_path),
        "episode_count": len(ids),
        "worker_count": 8,
        "physical_gpu_count": 4,
        "workers_per_gpu": 2,
    }, indent=2))


if __name__ == "__main__":
    main()
