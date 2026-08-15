#!/usr/bin/env bash
# Two-GPU archived-TTTRL baseline matched to test/run_test.sh.
# Usage: GPU_IDS="0 1" bash ori_run_test.sh

set -euo pipefail

WORKSPACE=/nvme-ssd2/lixintao/projects/test-time-training-vln
cd "$WORKSPACE"

export PYTHONPATH=/nvme-ssd2/lixintao/projects/NaVIDA:${PYTHONPATH:-}
PYTHON=/nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python
CONFIG=/nvme-ssd2/lixintao/projects/NaVIDA/config/vln_r2r.yaml
MODEL=/nvme-ssd2/lixintao/projects/NaVIDA/result/navida_train_smoke
SOURCE_DATASET=/nvme-ssd1/zwy/data/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
RESULT_ROOT="$WORKSPACE/results/r2r_ori_tttrl_smoke_2gpu"

# Exactly the episodes selected by test/run_test.sh:
# split-num=2, split-id=0/1, and max-episodes=1.
EPISODE_IDS=("1" "2")
GPU_IDS_STRING=${GPU_IDS:-"0 1"}
read -r -a GPU_ID_LIST <<< "$GPU_IDS_STRING"

if [[ ${#GPU_ID_LIST[@]} -ne 2 ]]; then
  echo "GPU_IDS must name exactly two idle physical GPU IDs; got: $GPU_IDS_STRING" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_DATASET" ]]; then
  echo "R2R source dataset not found: $SOURCE_DATASET" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT"
printf '%s
'   "archived_trainer=ori_ttrl-src/train/tttrl_main.py"   "reference_script=test/run_test.sh"   "episode_ids=${EPISODE_IDS[*]}"   "gpu_ids=${GPU_IDS_STRING}"   "num_generations=3"   "temperature=0.3"   "learning_rate=2e-6"   "weight_decay=0.0"   "max_grad_norm=1.0"   "lora_r=8"   "lora_alpha=8"   "lora_dropout=0.05"   "seed=41" > "$RESULT_ROOT/benchmark_controls.txt"

pids=()
for SPLIT_ID in 0 1; do
  SPLIT_DIR="$RESULT_ROOT/split_${SPLIT_ID}"
  mkdir -p "$SPLIT_DIR"
  CUDA_VISIBLE_DEVICES="${GPU_ID_LIST[$SPLIT_ID]}" "$PYTHON"     ori_tttrl_compat_runner.py       --source-dataset "$SOURCE_DATASET"       --episode-ids "${EPISODE_IDS[$SPLIT_ID]}"       --result-path "$SPLIT_DIR"       --metadata-path "$SPLIT_DIR/benchmark_metadata.json"       --       --model_name_or_path "$MODEL"       --dataset_path "$SPLIT_DIR/episodes.json.gz"       --simulator_config_path "$CONFIG"       --log_save_path "$SPLIT_DIR"       --output_dir "$SPLIT_DIR/trainer_output"       --per_device_train_batch_size 1       --num_train_epochs 1       --learning_rate 2e-6       --weight_decay 0.0       --max_grad_norm 1.0       --optim adamw_torch       --num_generations 3       --temperature 0.3       --lora_r 8       --lora_alpha 8       --lora_dropout 0.05       --seed 41       --bf16 True       --gradient_checkpointing False       --dataloader_num_workers 0       --save_strategy no       --logging_strategy steps       --logging_steps 1       --report_to none       >"$RESULT_ROOT/split_${SPLIT_ID}.out" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "split_${index} failed; inspect $RESULT_ROOT/split_${index}.out and benchmark_metadata.json" >&2
    status=1
  fi
done

if [[ $status -ne 0 ]]; then
  exit "$status"
fi
echo "Archived TTTRL two-GPU benchmark complete: $RESULT_ROOT"
