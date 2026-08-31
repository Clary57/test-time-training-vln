import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


selection = json.loads((ROOT / "selection1839.json").read_text())
ids = selection["episode_ids"]
assert len(ids) == len(set(ids)) == 1839
assert sha256(ROOT / "selection1839.json") == "1e704f18c79efbf6a6dc44001c19ab0990eb20a952854a8b5db55b9e0f208a6e"
launch = (ROOT / "launch.sh").read_text()
for required in (
    "eval_official_subset.py", "--max-action-history 200", "--num-generations 1",
    "SPLIT_NUM=8", "assert_gpu_idle", "CUDA_VISIBLE_DEVICES",
):
    assert required in launch, required
assert "eval_ttt.py" not in launch
print("PASS: official frozen full preparation")
