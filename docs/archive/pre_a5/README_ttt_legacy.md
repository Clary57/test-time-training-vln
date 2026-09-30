# Episode-local step-wise TTRL evaluation

`eval_ttt.py` is deliberately a **single-process** evaluator.  Run one ordinary
Python process per visible GPU; do not use `torchrun`, DDP, DeepSpeed, or
Accelerate.  Each process receives a disjoint deterministic episode shard,
restores the same base LoRA at the start of every episode, and writes its own
result directory.

## Run the unit tests

The tests use a tiny CPU fake model and do not load a Habitat scene or a model
checkpoint.  Use the `navida` environment, which contains PyTorch,
Transformers, PEFT, and FastDTW:

```bash
/nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python -m unittest \
  ttrl-src/tests/test_trajectory_consensus_updater.py
```


## Real two-episode Habitat smoke test

The local NaVIDA checkout provides a compatible R2R config and a small smoke
checkpoint.  The evaluator needs NaVIDA's `habitat_extensions` on
`PYTHONPATH`.  `--early-stop-steps 0` permits the first model decision (and
therefore one `adapt_step`) in each episode, then forces a stop on the next
loop iteration.  It is a protocol smoke test, not a navigation-quality result.

```bash
export PYTHONPATH=/nvme-ssd2/lixintao/projects/NaVIDA:${PYTHONPATH}
PYTHON=/nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python
CUDA_VISIBLE_DEVICES=0 "$PYTHON" ttrl-src/eval/eval_ttt.py \
  --exp-config /nvme-ssd2/lixintao/projects/NaVIDA/config/vln_r2r.yaml \
  --model-path /nvme-ssd2/lixintao/projects/NaVIDA/result/navida_train_smoke \
  --result-path /path/to/results/ttt_smoke \
  --split-num 1 --split-id 0 \
  --max-episodes 2 --early-stop-steps 0 \
  --num-generations 3 --max-action-history 10 --seed 41
```

The resulting directory must contain `episode_manifest.json`, two
`stats_*.json` records, and two `steps_*.json` records.  For a one-shard smoke
run, validate it with the same strict aggregator:

```bash
"$PYTHON" ttrl-src/eval/aggregate_ttt_results.py \
  --result-dirs /path/to/results/ttt_smoke
```

## Launch independent GPU shards

Set `GPU_IDS` to the actual GPUs visible to the shell.  The example supports 8
GPUs when they are visible, but it also works with any positive number of GPUs.
Every process must use a distinct `--result-path`.

```bash
PYTHON=/nvme-ssd2/lixintao/miniconda3/envs/navida/bin/python
CONFIG=/path/to/eval_config.yaml
MODEL_PATH=/path/to/navida_checkpoint
LORA_PATH=/path/to/initial_lora   # optional; omit --lora-path if unused
RESULT_ROOT=/path/to/results/ttt_eval
GPU_IDS=0,1,2,3,4,5,6,7

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
for SPLIT_ID in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$SPLIT_ID]}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" ttrl-src/eval/eval_ttt.py \
    --exp-config "$CONFIG" \
    --model-path "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --result-path "$RESULT_ROOT/split_${SPLIT_ID}" \
    --split-num "${#GPUS[@]}" \
    --split-id "$SPLIT_ID" \
    --num-generations 8 \
    --learning-rate 2e-6 \
    --weight-decay 0.0 \
    --max-grad-norm 1.0 \
    --temperature 0.3 \
    --seed 41 \
    >"$RESULT_ROOT/split_${SPLIT_ID}.out" 2>&1 &
done
wait
```

`CUDA_VISIBLE_DEVICES` exposes one physical GPU as logical `cuda:0` to each
process.  `eval_ttt.py` rejects a distributed process group and `WORLD_SIZE >
1`; processes neither synchronize nor wait for one another.

## Strictly aggregate all shards

The evaluator writes `episode_manifest.json` before resume filtering.  Each
manifest records both its assigned IDs and the full expected union for the
chosen split count (and optional smoke limit).  The aggregator reads every raw
`stats_<episode_id>.json`, rejects missing or duplicate episodes—including an
entire absent shard—and recomputes metrics from episode records rather than
averaging shard means.

```bash
"$PYTHON" ttrl-src/eval/aggregate_ttt_results.py \
  --result-dirs "$RESULT_ROOT"/split_* \
  --output "$RESULT_ROOT/summary.json"
```

A resumed shard is valid only when its manifest still matches a complete set of
`stats_*.json` records.  Delete or move stale result directories before changing
split count, dataset order, model, or seed.
