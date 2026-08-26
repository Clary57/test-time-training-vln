"""Pure helpers for deterministic episode subset selection and sharding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _normalize_requested_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("episode_ids entries must be strings or integers")
    return str(value)


def load_episode_ids_file(path: str | Path) -> tuple[list[str], dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if isinstance(payload, list):
        episode_ids = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, Mapping) and isinstance(payload.get("episode_ids"), list):
        episode_ids = payload["episode_ids"]
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("episode-ids-file metadata must be an object")
        metadata = dict(raw_metadata)
    else:
        raise ValueError("episode-ids-file must be a JSON list or an object with episode_ids")
    normalized = [_normalize_requested_id(value) for value in episode_ids]
    if not normalized:
        raise ValueError("episode_ids must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("episode_ids must not contain duplicates")
    return normalized, metadata, hashlib.sha256(raw).hexdigest()


def select_episodes_in_file_order(episodes: Sequence[Any], episode_ids: Sequence[str]) -> list[Any]:
    by_id = {str(episode.episode_id): episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("dataset contains duplicate episode IDs")
    unknown = [episode_id for episode_id in episode_ids if episode_id not in by_id]
    if unknown:
        raise ValueError(f"episode IDs not found in dataset: {unknown}")
    return [by_id[episode_id] for episode_id in episode_ids]


def shard_episodes(episodes: Sequence[Any], split_num: int, split_id: int) -> list[Any]:
    if split_num < 1 or not 0 <= split_id < split_num:
        raise ValueError("split-id must be in [0, split-num)")
    return list(episodes[split_id::split_num])
