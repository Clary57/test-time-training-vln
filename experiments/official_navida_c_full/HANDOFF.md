# Official NaVIDA frozen full val_unseen

Runs all 1,839 locked R2R VLN-CE val_unseen episodes with the already validated
official-lineage frozen evaluator. Eight official shard/process boundaries are
preserved. The caller supplies a comma-separated list of one or more idle physical
GPU IDs; shards are scheduled in deterministic waves with at most one evaluator per
GPU. No GPU is selected automatically and failed shards are not retried.

This may run concurrently with `official_navida_ttrl_a128` only when the two launches
receive disjoint GPU lists and independent result roots.
