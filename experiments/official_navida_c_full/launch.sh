#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RESULT_ROOT GPU_IDS_CSV" >&2
  exit 2
fi

RESULT_ROOT=$1
IFS=',' read -r -a GPUS <<< "$2"
PROJECT=/hdd3/wangziyu/code/test-time-training-vln
NAVIDA_ROOT=/hdd3/wangziyu/code/Self-TTRL4VLN/NaVIDA
PYTHON=/hdd3/wangziyu/envs/navida/bin/python
EVALUATOR=$PROJECT/ttrl-src/eval/eval_official_subset.py
CONFIG=$NAVIDA_ROOT/config/vln_r2r.yaml
MODEL=/hdd3/wangziyu/models/NaVIDA
SELECTION=$PROJECT/experiments/official_navida_c_full/selection1839.json
EXPECTED_SELECTION_SHA=1e704f18c79efbf6a6dc44001c19ab0990eb20a952854a8b5db55b9e0f208a6e
EXPECTED_CONFIG_SHA=33f5361af8687f6852cc83f83c2a49e98be848acf501a4ad0907de181d15bf38
SPLIT_NUM=8

[[ ${#GPUS[@]} -ge 1 && ${#GPUS[@]} -le 8 ]] || { echo "provide 1-8 GPU IDs" >&2; exit 1; }
[[ $(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l) -eq ${#GPUS[@]} ]] || { echo "GPU IDs must be unique" >&2; exit 1; }
[[ ! -e "$RESULT_ROOT" ]] || { echo "result root already exists: $RESULT_ROOT" >&2; exit 1; }
[[ $(git -C "$PROJECT" branch --show-current) == codex/official-navida-ttrl-and-cfull ]] || { echo "wrong Git branch" >&2; exit 1; }
[[ -z $(git -C "$PROJECT" status --porcelain) ]] || { echo "Git worktree is not clean" >&2; exit 1; }
[[ $(sha256sum "$SELECTION" | awk '{print $1}') == "$EXPECTED_SELECTION_SHA" ]] || { echo "selection hash mismatch" >&2; exit 1; }
[[ $(sha256sum "$CONFIG" | awk '{print $1}') == "$EXPECTED_CONFIG_SHA" ]] || { echo "config hash mismatch" >&2; exit 1; }

assert_gpu_idle() {
  local gpu=$1 compute_count free_mib
  compute_count=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [[ $compute_count -eq 0 ]] || { echo "GPU $gpu has compute processes" >&2; return 1; }
  [[ $free_mib -ge 45000 ]] || { echo "GPU $gpu has only ${free_mib} MiB free" >&2; return 1; }
}
for gpu in "${GPUS[@]}"; do assert_gpu_idle "$gpu"; done

mkdir -p "$RESULT_ROOT"
cp "$SELECTION" "$RESULT_ROOT/selection1839.json"
cp "$PROJECT/experiments/official_navida_c_full/analysis_plan.json" "$RESULT_ROOT/analysis_plan.json"
cp "$PROJECT/experiments/official_navida_c_full/protocol_manifest.json" "$RESULT_ROOT/protocol_manifest.json"
git -C "$PROJECT" status --short --branch > "$RESULT_ROOT/git_status_at_launch.txt"
git -C "$PROJECT" rev-parse HEAD > "$RESULT_ROOT/git_head_at_launch.txt"
printf '%s\n' "${GPUS[@]}" > "$RESULT_ROOT/physical_gpus.txt"

for ((batch=0; batch<SPLIT_NUM; batch+=${#GPUS[@]})); do
  pids=()
  for ((slot=0; slot<${#GPUS[@]} && batch+slot<SPLIT_NUM; slot++)); do
    split_id=$((batch+slot)); gpu=${GPUS[$slot]}; assert_gpu_idle "$gpu"
    worker_root=$RESULT_ROOT/worker_$split_id; mkdir "$worker_root"
    command=(env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR"
      --exp-config "$CONFIG" --split-num "$SPLIT_NUM" --split-id "$split_id"
      --forward-distance 25 --turn-angle 15 --resolution-ratio 0.5
      --max-action-history 200 --model-path "$MODEL" --num-generations 1
      --episode-ids-file "$RESULT_ROOT/selection1839.json" --result-path "$worker_root")
    printf '%q ' "${command[@]}" > "$worker_root/command.txt"; printf '\n' >> "$worker_root/command.txt"
    printf '%s\n' "$gpu" > "$worker_root/physical_gpu.txt"
    (
      cd "$NAVIDA_ROOT"; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NAVIDA_ROOT:$PROJECT/ttrl-src"
      set +e; "${command[@]}" > "$worker_root/stdout.log" 2> "$worker_root/stderr.log"
      rc=$?; printf '%s\n' "$rc" > "$worker_root/eval_exit_code.txt"; exit "$rc"
    ) & pids+=("$!")
  done
  failed=0; for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  if [[ $failed -ne 0 ]]; then touch "$RESULT_ROOT/FAILED"; exit 1; fi
done
touch "$RESULT_ROOT/SUCCESS"
