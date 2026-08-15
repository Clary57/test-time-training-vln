"""Strict aggregation for independent episode-local TTRL evaluation shards.

Each evaluator writes ``episode_manifest.json`` before resume filtering and one
``log/stats_<episode_id>.json`` file per finished episode.  This script merges
raw episode metrics, never averages per-shard means, and rejects duplicate or
missing IDs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

METRIC_KEYS = (
    "success",
    "spl",
    "distance_to_goal",
    "path_length",
    "oracle_success",
    "ndtw",
)
MANIFEST_NAME = "episode_manifest.json"


def _load_manifest(result_dir: Path) -> Dict[str, List[str]]:
    manifest_path = result_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Missing shard manifest: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("episode_ids"), list):
        raise ValueError(f"Invalid shard manifest: {manifest_path}")
    episode_ids = [str(episode_id) for episode_id in manifest["episode_ids"]]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f"Duplicate episode IDs inside manifest: {manifest_path}")
    expected_ids = manifest.get("expected_episode_ids")
    if expected_ids is not None:
        if not isinstance(expected_ids, list):
            raise ValueError(f"Invalid expected episode IDs: {manifest_path}")
        expected_ids = [str(episode_id) for episode_id in expected_ids]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError(f"Duplicate expected episode IDs: {manifest_path}")
        if not set(episode_ids).issubset(expected_ids):
            raise ValueError(f"Shard assignment is outside expected IDs: {manifest_path}")
    return {"episode_ids": episode_ids, "expected_episode_ids": expected_ids}


def _finite_metric(value: object, metric: str, stats_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{stats_path}: metric {metric!r} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{stats_path}: metric {metric!r} is not finite")
    return value


def aggregate_result_dirs(result_dirs: Sequence[Path]) -> Dict[str, object]:
    """Load, validate, and aggregate disjoint evaluator result directories."""
    if not result_dirs:
        raise ValueError("At least one result directory is required")

    assigned_ids: Set[str] = set()
    shared_expected_ids: Set[str] | None = None
    stats_by_id: Dict[str, Mapping[str, object]] = {}
    source_by_id: Dict[str, Path] = {}
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        manifest = _load_manifest(result_dir)
        manifest_ids = manifest["episode_ids"]
        overlap = assigned_ids.intersection(manifest_ids)
        if overlap:
            raise ValueError(f"Episode IDs assigned to multiple manifests: {sorted(overlap)}")
        assigned_ids.update(manifest_ids)
        manifest_expected = manifest["expected_episode_ids"]
        if manifest_expected is not None:
            manifest_expected_set = set(manifest_expected)
            if shared_expected_ids is None:
                shared_expected_ids = manifest_expected_set
            elif shared_expected_ids != manifest_expected_set:
                raise ValueError("Shard manifests disagree on the global expected episode IDs")

        log_dir = result_dir / "log"
        if not log_dir.is_dir():
            raise ValueError(f"Missing log directory: {log_dir}")
        for stats_path in sorted(log_dir.glob("stats_*.json")):
            with stats_path.open() as handle:
                stats = json.load(handle)
            if not isinstance(stats, Mapping) or "id" not in stats:
                raise ValueError(f"Invalid stats record: {stats_path}")
            episode_id = str(stats["id"])
            if episode_id in stats_by_id:
                raise ValueError(
                    f"Duplicate result for episode {episode_id}: "
                    f"{source_by_id[episode_id]} and {stats_path}"
                )
            stats_by_id[episode_id] = stats
            source_by_id[episode_id] = stats_path

    expected_ids = shared_expected_ids if shared_expected_ids is not None else assigned_ids
    actual_ids = set(stats_by_id)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError("Episode coverage validation failed: " + "; ".join(details))

    metric_values: Dict[str, List[float]] = {metric: [] for metric in METRIC_KEYS}
    for episode_id in sorted(actual_ids):
        stats = stats_by_id[episode_id]
        for metric in METRIC_KEYS:
            if metric in stats:
                metric_values[metric].append(_finite_metric(stats[metric], metric, source_by_id[episode_id]))

    metrics = {
        metric: {
            "mean": sum(values) / len(values),
            "count": len(values),
        }
        for metric, values in metric_values.items()
        if values
    }
    return {
        "episode_count": len(actual_ids),
        "episode_ids": sorted(actual_ids),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="per-split result directories, each containing a manifest and log directory",
    )
    parser.add_argument("--output", type=Path, help="optional JSON path for the merged summary")
    args = parser.parse_args()

    summary = aggregate_result_dirs(args.result_dirs)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
