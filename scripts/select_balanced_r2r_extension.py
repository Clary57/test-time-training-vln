#!/usr/bin/env python3
"""Extend a prior representative R2R selection to balanced length classes."""

import argparse
import json
from collections import Counter
from pathlib import Path

from select_representative_r2r_episodes import load_json_gz, select, select_balanced_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--old-selection", type=Path, required=True)
    parser.add_argument("--smoke-selection", type=Path, required=True)
    parser.add_argument("--selection72", type=Path, required=True)
    parser.add_argument("--additional52", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.seed != 41:
        raise ValueError("balanced representative-72 is fixed to seed=41")
    old = json.loads(args.old_selection.read_text())
    old_ids = [str(value) for value in old["episode_ids"]]
    smoke_ids = [str(value) for value in json.loads(args.smoke_selection.read_text())["episode_ids"]]
    _, records = select(args.dataset, args.gt, 20, args.seed)
    old_meta = {item["episode_id"]: item for item in old["metadata"]["selected_episodes"]}
    old_counts = Counter(old_meta[episode_id]["path_length_bin"] for episode_id in old_ids)
    targets = {"short": 24 - old_counts["short"], "middle": 24 - old_counts["middle"], "long": 24 - old_counts["long"]}
    new_records, class_audit = select_balanced_extension(
        records, old_ids, targets, args.seed, smoke_ids
    )
    new_ids = [item["episode_id"] for item in new_records]
    final_ids = old_ids + new_ids
    if len(old_ids) != 20 or len(new_ids) != 52 or len(final_ids) != len(set(final_ids)) != 72:
        raise ValueError("final selection cardinality validation failed")
    final_records = []
    for episode_id in old_ids:
        item = dict(old_meta[episode_id])
        item.update(source="old", selection_reason="reused validated representative-20 formal result")
        final_records.append(item)
    for item in new_records:
        final_records.append(dict(item, source="new"))
    final_counts = Counter(item["path_length_bin"] for item in final_records)
    if final_counts != {"short": 24, "middle": 24, "long": 24}:
        raise ValueError(f"final class balance validation failed: {final_counts}")
    metadata = {
        "sample_size": 72, "seed": args.seed, "dataset_split": "val_unseen",
        "dataset_sha256": old["metadata"]["dataset_sha256"], "gt_sha256": old["metadata"]["gt_sha256"],
        "path_length_definition": old["metadata"]["path_length_definition"],
        "selection_algorithm": "reuse formal representative-20; per-class capacity-proportional largest remainder; seeded scene/class draws",
        "selection_uses_model_results": False, "old_count": 20, "new_count": 52,
        "class_counts": dict(final_counts), "selected_episodes": final_records,
    }
    selection72 = {"episode_ids": final_ids, "metadata": metadata}
    additional = {"episode_ids": new_ids, "metadata": {**metadata, "sample_size": 52, "selected_episodes": new_records}}
    audit = {
        "old_class_counts": dict(old_counts), "additional_targets": targets,
        "smoke_ids_excluded": smoke_ids,
        "smoke_ids_already_in_formal_old20": sorted(set(smoke_ids) & set(old_ids)),
        "final_class_counts": dict(final_counts), "class_allocation": class_audit,
        "integrity": {"old_unique": len(set(old_ids)) == 20, "new_unique": len(set(new_ids)) == 52,
                      "old_new_disjoint": not set(old_ids) & set(new_ids), "final_unique": len(set(final_ids)) == 72,
                      "all_in_val_unseen": set(final_ids) <= {item["episode_id"] for item in records},
                      "selection_uses_model_results": False},
    }
    outputs = (args.selection72, args.additional52, args.audit, args.table)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an extension selection output")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.selection72.write_text(json.dumps(selection72, indent=2) + "\n")
    args.additional52.write_text(json.dumps(additional, indent=2) + "\n")
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["episode_id\tsource\tscene_id\treference_path_points\tlength_class\tselection_reason"]
    lines += ["\t".join(map(str, (item["episode_id"], item["source"], item["scene_id"], item["path_length"], item["path_length_bin"], item["selection_reason"]))) for item in final_records]
    args.table.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
