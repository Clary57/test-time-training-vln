# Official-protocol TTRL A128

Runs the locked confirmatory128 IDs through the existing episode-local TTRL updater
while matching the validated official frozen path's prompt/image-input construction,
two action chunks, history 200, model, config, and eight shard boundaries.

The caller supplies a comma-separated list of idle physical GPU IDs. Shards run in
deterministic waves with at most one evaluator per GPU. Use a result root and GPU list
that are disjoint from the full frozen baseline launch.

The primary paired comparator is the completed official frozen C0-128 (72/128), not
the historical non-canonical frozen run. This A/C comparison measures the complete
TTRL stack and does not by itself isolate online updates from consensus selection.
