#!/usr/bin/env python3
"""Select a deterministic scene/path-length-stratified R2R pilot sample."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def distribution(values):
    return {
        "min": float(min(values)), "p25": percentile(values, 0.25),
        "median": float(median(values)), "p75": percentile(values, 0.75),
        "max": float(max(values)),
    }


def allocate_scene_counts(groups, sample_size):
    scenes = sorted(groups)
    if len(scenes) > sample_size:
        raise ValueError("scene count exceeds sample size; cannot cover every scene")
    allocation = {scene: 1 for scene in scenes}
    remaining = sample_size - len(scenes)
    total = sum(len(groups[scene]) for scene in scenes)
    quotas = {scene: remaining * len(groups[scene]) / total for scene in scenes}
    floors = {scene: math.floor(quotas[scene]) for scene in scenes}
    for scene in scenes:
        allocation[scene] += floors[scene]
    leftover = remaining - sum(floors.values())
    order = sorted(scenes, key=lambda scene: (-(quotas[scene] - floors[scene]), scene))
    for scene in order[:leftover]:
        allocation[scene] += 1
    if any(allocation[scene] > len(groups[scene]) for scene in scenes):
        raise ValueError("scene allocation exceeds available episodes")
    return allocation


def bin_scene_episodes(episodes):
    ordered = sorted(episodes, key=lambda item: (item["path_length"], item["episode_id"]))
    bins = {"short": [], "middle": [], "long": []}
    labels = ("short", "middle", "long")
    for rank, episode in enumerate(ordered):
        label = labels[min(2, rank * 3 // len(ordered))]
        episode["path_length_bin"] = label
        bins[label].append(episode)
    return bins


def select(dataset_path: Path, gt_path: Path, sample_size: int, seed: int):
    payload = load_json_gz(dataset_path)
    gt = load_json_gz(gt_path)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset must contain a non-empty episodes list")
    records = []
    seen = set()
    definitions = set()
    for episode in episodes:
        episode_id = str(episode.get("episode_id"))
        if episode_id in seen:
            raise ValueError(f"duplicate dataset episode ID: {episode_id}")
        seen.add(episode_id)
        if episode_id not in gt:
            raise ValueError(f"episode {episode_id} has no corresponding GT")
        reference_path = episode.get("reference_path")
        if isinstance(reference_path, list) and reference_path:
            path_length = len(reference_path)
            definition = "dataset.reference_path point count (GT locations fallback)"
        else:
            locations = gt[episode_id].get("locations")
            if not isinstance(locations, list) or not locations:
                raise ValueError(f"episode {episode_id} has neither reference_path nor GT locations")
            path_length = len(locations)
            definition = "GT locations point count (dataset.reference_path unavailable)"
        definitions.add(definition)
        instruction = episode.get("instruction", {})
        text = instruction.get("instruction_text")
        if not isinstance(text, str):
            raise ValueError(f"episode {episode_id} has no instruction text")
        records.append({
            "episode_id": episode_id, "scene_id": episode.get("scene_id"),
            "path_length": path_length, "instruction_length": len(text.split()),
        })
    if any(not isinstance(record["scene_id"], str) for record in records):
        raise ValueError("every episode must have a string scene_id")
    if sample_size != 20 or seed != 41:
        raise ValueError("this pilot selector is fixed to sample-size=20 and seed=41")
    groups = defaultdict(list)
    for record in records:
        groups[record["scene_id"]].append(record)
    allocation = allocate_scene_counts(groups, sample_size)
    selected_by_scene = {}
    for scene_index, scene in enumerate(sorted(groups)):
        bins = bin_scene_episodes(groups[scene])
        for label, items in bins.items():
            stable_seed = int(hashlib.sha256(f"{seed}:{scene}:{label}".encode()).hexdigest()[:16], 16)
            random.Random(stable_seed).shuffle(items)
        chosen = []
        base_labels = ("short", "middle", "long")
        offset = scene_index % len(base_labels)
        labels = base_labels[offset:] + base_labels[:offset]
        while len(chosen) < allocation[scene]:
            progressed = False
            for label in labels:
                if bins[label] and len(chosen) < allocation[scene]:
                    chosen.append(bins[label].pop())
                    progressed = True
            if not progressed:
                raise ValueError(f"unable to fill allocation for scene {scene}")
        selected_by_scene[scene] = chosen
    selected = []
    max_scene_selection = max(map(len, selected_by_scene.values()))
    for index in range(max_scene_selection):
        for scene in sorted(selected_by_scene):
            if index < len(selected_by_scene[scene]):
                selected.append(selected_by_scene[scene][index])
    selected_ids = [item["episode_id"] for item in selected]
    if len(selected_ids) != 20 or len(set(selected_ids)) != 20 or not set(selected_ids) <= seen:
        raise ValueError("selection validation failed")
    path_definition = next(iter(definitions)) if len(definitions) == 1 else "; mixed: " + "; ".join(sorted(definitions))
    result = {
        "episode_ids": selected_ids,
        "metadata": {
            "sample_size": sample_size, "seed": seed, "dataset_split": "val_unseen",
            "dataset_sha256": sha256(dataset_path), "gt_sha256": sha256(gt_path),
            "path_length_definition": path_definition,
            "selection_algorithm": "scene coverage + largest remainder; per-scene path-length tertiles; seeded round-robin",
            "population_episode_count": len(records), "selected_episodes": selected,
            "selection_uses_model_results": False,
        },
    }
    return result, records


def make_report(result, population):
    selected = result["metadata"]["selected_episodes"]
    all_scenes = Counter(item["scene_id"] for item in population)
    selected_scenes = Counter(item["scene_id"] for item in selected)
    return {
        "population_episode_count": len(population), "sample_episode_count": len(selected),
        "population_scene_count": len(all_scenes), "sample_scene_count": len(selected_scenes),
        "scene_coverage": len(selected_scenes) / len(all_scenes),
        "scenes": {scene: {"population": all_scenes[scene], "selected": selected_scenes[scene]} for scene in sorted(all_scenes)},
        "path_length": {"population": distribution([x["path_length"] for x in population]), "sample": distribution([x["path_length"] for x in selected])},
        "instruction_length": {"population": distribution([x["instruction_length"] for x in population]), "sample": distribution([x["instruction_length"] for x in selected])},
        "path_length_bins": dict(sorted(Counter(x["path_length_bin"] for x in selected).items())),
    }


def render_report(report):
    lines = [
        f"Episodes: population={report['population_episode_count']} selected={report['sample_episode_count']}",
        f"Scenes: population={report['population_scene_count']} selected={report['sample_scene_count']} coverage={report['scene_coverage']:.3f}",
        "Scenes (population/selected):",
    ]
    lines.extend(f"  {scene}: {counts['population']}/{counts['selected']}" for scene, counts in report["scenes"].items())
    for name in ("path_length", "instruction_length"):
        lines.append(f"{name}: population={report[name]['population']} sample={report[name]['sample']}")
    lines.append(f"Path bins: {report['path_length_bins']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-text", type=Path, required=True)
    args = parser.parse_args()
    result, population = select(args.dataset, args.gt, args.sample_size, args.seed)
    report = make_report(result, population)
    for path in (args.output, args.report_json, args.report_text):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    args.report_text.write_text(render_report(report))


if __name__ == "__main__":
    main()
