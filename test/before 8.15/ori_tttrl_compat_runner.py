#!/usr/bin/env python3
"""Compatibility launcher and wall-clock recorder for the archived TTTRL trainer."""

from __future__ import annotations

import argparse
import base64
import gzip
import io
from importlib.machinery import ModuleSpec
import json
import runpy
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "ori_ttrl-src"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer_args = list(args.trainer_args)
    if trainer_args[:1] == ["--"]:
        trainer_args = trainer_args[1:]
    if not trainer_args:
        parser.error("trainer arguments are required after --")
    return args, trainer_args


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def prepare_subset(source: Path, ids: list[str], destination: Path) -> None:
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    by_id = {str(episode["episode_id"]): episode for episode in data["episodes"]}
    missing = [episode_id for episode_id in ids if episode_id not in by_id]
    if missing:
        raise ValueError(f"Episode IDs not found in {source}: {missing}")
    data["episodes"] = [by_id[episode_id] for episode_id in ids]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        json.dump(data, handle)


def install_compatibility_shims() -> None:
    if not ARCHIVE.is_dir():
        raise FileNotFoundError(f"Archived source is missing: {ARCHIVE}")
    sys.path.insert(0, str(ARCHIVE))

    src = types.ModuleType("src")
    src.__path__ = [str(ARCHIVE)]  # type: ignore[attr-defined]
    sys.modules["src"] = src

    def encode_image_base64(image: Any) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    vllm = types.ModuleType("vllm")
    vllm.__spec__ = ModuleSpec("vllm", loader=None, is_package=True)
    vllm.__path__ = []  # type: ignore[attr-defined]
    multimodal = types.ModuleType("vllm.multimodal")
    utils = types.ModuleType("vllm.multimodal.utils")
    utils.encode_image_base64 = encode_image_base64  # type: ignore[attr-defined]
    vllm.multimodal = multimodal  # type: ignore[attr-defined]
    multimodal.utils = utils  # type: ignore[attr-defined]
    sys.modules.update({
        "vllm": vllm,
        "vllm.multimodal": multimodal,
        "vllm.multimodal.utils": utils,
    })


def main() -> None:
    args, trainer_args = parse_args()
    ids = [item.strip() for item in args.episode_ids.split(",") if item.strip()]
    if not ids:
        raise ValueError("--episode-ids must contain at least one ID")

    result_path = args.result_path.resolve()
    subset_path = result_path / "episodes.json.gz"
    metadata: dict[str, Any] = {
        "archive": str(ARCHIVE),
        "compatibility_shims": [
            "src package alias for archived layout",
            "local JPEG base64 helper replacing archived vLLM-only utility import",
        ],
        "episode_ids": ids,
        "source_dataset": str(args.source_dataset),
        "subset_dataset": str(subset_path),
        "trainer_args": trainer_args,
        "started_at_unix": time.time(),
        "status": "running",
    }
    write_json(args.metadata_path, metadata)

    started = time.perf_counter()
    try:
        prepare_subset(args.source_dataset, ids, subset_path)
        install_compatibility_shims()
        sys.argv = [str(ARCHIVE / "train" / "tttrl_main.py"), *trainer_args]
        runpy.run_path(str(ARCHIVE / "train" / "tttrl_main.py"), run_name="__main__")
        metadata["status"] = "completed"
        train_metrics = result_path / "trainer_output" / "train_results.json"
        if train_metrics.is_file():
            metadata["trainer_metrics"] = json.loads(train_metrics.read_text(encoding="utf-8"))
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = repr(error)
        metadata["traceback"] = traceback.format_exc()
        raise
    finally:
        metadata["wall_clock_seconds"] = time.perf_counter() - started
        metadata["finished_at_unix"] = time.time()
        write_json(args.metadata_path, metadata)


if __name__ == "__main__":
    main()
