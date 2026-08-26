#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RESULT_ROOT GPU0 GPU1 GPU2 GPU3" >&2
  exit 2
fi

RESULT_ROOT=$1
shift
GPUS=("$@")

PROJECT=/hdd3/wangziyu/code/test-time-training-vln
NAVIDA_ROOT=/hdd3/wangziyu/code/Self-TTRL4VLN/NaVIDA
PYTHON=/hdd3/wangziyu/envs/navida/bin/python
EVALUATOR=$PROJECT/ttrl-src/eval/eval_official_subset.py
CONFIG=$NAVIDA_ROOT/config/vln_r2r.yaml
MODEL=/hdd3/wangziyu/models/NaVIDA
SELECTION=$PROJECT/experiments/official_navida_c128/selection128.json
EXPECTED_SELECTION_SHA=e3d3c050d471ca86a253ca8238cd662a6208e72c10ed0bdb2248390329becfb6
EXPECTED_CONFIG_SHA=33f5361af8687f6852cc83f83c2a49e98be848acf501a4ad0907de181d15bf38
SPLIT_NUM=8

[[ ! -e "$RESULT_ROOT" ]] || { echo "result root already exists: $RESULT_ROOT" >&2; exit 1; }
[[ $(sha256sum "$SELECTION" | awk '{print $1}') == "$EXPECTED_SELECTION_SHA" ]] || {
  echo "selection hash mismatch" >&2; exit 1;
}
[[ $(sha256sum "$CONFIG" | awk '{print $1}') == "$EXPECTED_CONFIG_SHA" ]] || {
  echo "config hash mismatch" >&2; exit 1;
}
[[ ${#GPUS[@]} -eq 4 ]] || { echo "exactly four physical GPUs are required" >&2; exit 1; }
[[ $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) -eq 4 ]] || {
  echo "physical GPU IDs must be distinct" >&2; exit 1;
}

for gpu in "${GPUS[@]}"; do
  compute_count=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [[ $compute_count -eq 0 ]] || { echo "GPU $gpu has compute processes" >&2; exit 1; }
  [[ $free_mib -ge 45000 ]] || { echo "GPU $gpu has only ${free_mib} MiB free" >&2; exit 1; }
done

mkdir -p "$RESULT_ROOT"
cp "$SELECTION" "$RESULT_ROOT/selection128.json"
cp "$PROJECT/experiments/official_navida_c128/analysis_plan.json" "$RESULT_ROOT/analysis_plan.json"
cp "$PROJECT/experiments/official_navida_c128/protocol_manifest.json" "$RESULT_ROOT/protocol_manifest.json"
git -C "$PROJECT" status --short --branch > "$RESULT_ROOT/git_status_at_launch.txt"
git -C "$PROJECT" rev-parse HEAD > "$RESULT_ROOT/git_head_at_launch.txt"

worker_pids=()
for split_id in $(seq 0 $((SPLIT_NUM - 1))); do
  gpu=${GPUS[$((split_id / 2))]}
  worker_root=$RESULT_ROOT/worker_$split_id
  mkdir "$worker_root"
  command=(
    env CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" "$EVALUATOR"
    --exp-config "$CONFIG"
    --split-num "$SPLIT_NUM"
    --split-id "$split_id"
    --forward-distance 25
    --turn-angle 15
    --resolution-ratio 0.5
    --max-action-history 200
    --model-path "$MODEL"
    --num-generations 1
    --episode-ids-file "$RESULT_ROOT/selection128.json"
    --result-path "$worker_root"
  )
  printf '%q ' "${command[@]}" > "$worker_root/command.txt"
  printf '\n' >> "$worker_root/command.txt"
  printf '%s\n' "$gpu" > "$worker_root/physical_gpu.txt"
  (
    cd "$NAVIDA_ROOT"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$NAVIDA_ROOT:$PROJECT/ttrl-src"
    set +e
    "${command[@]}" > "$worker_root/stdout.log" 2> "$worker_root/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$worker_root/eval_exit_code.txt"
    exit "$rc"
  ) &
  worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
[[ $failed -eq 0 ]] || { touch "$RESULT_ROOT/FAILED"; exit 1; }
touch "$RESULT_ROOT/SUCCESS"
