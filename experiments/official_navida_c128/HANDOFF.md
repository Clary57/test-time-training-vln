# Official NaVIDA frozen confirmatory128 preparation

This run uses the public `waynechu1021/NAVIDA` evaluator at commit
`b86316a78a2865c80b445ed5a8fcc08fc0243dfb`, with only deterministic ordered
episode-subset selection and provenance output added.

Protocol-critical official behavior remains unchanged:

- Transformers evaluator (`src/eval/eval.py` lineage)
- two parsed action chunks per model decision
- `max-action-history=200` from official `scripts/eval.sh`
- one generation, temperature 0.2, top-p 1.0, 512 new tokens, repetition penalty 1.05
- no LoRA and no adaptation
- eight evaluator shards, run as eight fresh Python processes serially on one GPU

The launch script requires a new result root and one explicitly selected, fully idle
physical GPU. It preserves the official eight process/seed boundaries but serializes
them to remove concurrency and reduce resource requirements. It does not select a GPU
automatically and does not retry.

Example for a future separately approved launch:

```bash
bash experiments/official_navida_c128/launch.sh \
  /hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_<timestamp> \
  GPU_ID
```

This preparation does not claim that the 128-episode SR equals the published full
1839-episode result. Its purpose is to determine whether the historical project
evaluator materially depressed the frozen baseline on the same locked episodes.
