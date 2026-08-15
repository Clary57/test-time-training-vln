cd /nvme-ssd2/lixintao/projects/test-time-training-vln

set -euo pipefail

export PYTHONPATH=/nvme-ssd2/lixintao/projects/NaVIDA:${PYTHONPATH:-}
PYTHON=/nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python
CONFIG=/nvme-ssd2/lixintao/projects/NaVIDA/config/vln_r2r.yaml
MODEL=/nvme-ssd2/lixintao/projects/NaVIDA/result/navida_train_smoke
RESULT_ROOT=/nvme-ssd2/lixintao/projects/test-time-training-vln/results/r2r_ttt_short_4gpu
mkdir -p "$RESULT_ROOT"

for SPLIT_ID in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$SPLIT_ID" "$PYTHON" ttrl-src/eval/eval_ttt.py \
    --exp-config "$CONFIG" \
    --model-path "$MODEL" \
    --result-path "$RESULT_ROOT/split_${SPLIT_ID}" \
    --split-num 4 \
    --split-id "$SPLIT_ID" \
    --max-episodes 2 \
    --early-stop-steps 10 \
    --num-generations 3 \
    --max-action-history 10 \
    --learning-rate 2e-6 \
    --max-grad-norm 1.0 \
    --temperature 0.3 \
    --seed 41 \
    >"$RESULT_ROOT/split_${SPLIT_ID}.out" 2>&1 &
done
wait