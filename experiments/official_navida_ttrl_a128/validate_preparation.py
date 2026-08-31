import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


selection = json.loads((ROOT / "selection128.json").read_text())
ids = selection["episode_ids"]
assert len(ids) == len(set(ids)) == 128
assert sha256(ROOT / "selection128.json") == "e3d3c050d471ca86a253ca8238cd662a6208e72c10ed0bdb2248390329becfb6"
launch = (ROOT / "launch.sh").read_text()
for required in (
    "eval_ttt.py", "--official-navida-protocol", "--max-action-history 200",
    "--num-generations 3", "--learning-rate 2e-6", "--temperature 0.3",
    "--stop-trace", "SPLIT_NUM=8", "assert_gpu_idle", "CUDA_VISIBLE_DEVICES",
):
    assert required in launch, required
assert "--disable-update" not in launch
assert "eval_official_subset.py" not in launch
print("PASS: official-protocol TTRL A128 preparation")
