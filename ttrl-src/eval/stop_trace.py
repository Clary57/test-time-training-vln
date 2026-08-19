"""Default-off, logging-only provenance for navigation atomic actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SUCCESS_DISTANCE_CONFIG_SOURCE = (
    "config.habitat.task.measurements.success.success_distance"
)


def resolve_success_distance(config: Any) -> float:
    """Resolve the exact threshold used by Habitat's Success measure."""
    try:
        value = config.habitat.task.measurements.success.success_distance
    except AttributeError as exc:
        raise ValueError(
            f"stop tracing requires {SUCCESS_DISTANCE_CONFIG_SOURCE}"
        ) from exc
    return float(value)


def selected_candidate_index(
    rewards: Iterable[float], valid: Iterable[bool]
) -> Optional[int]:
    """Mirror the updater's deterministic medoid index without choosing again."""
    rewards = list(rewards)
    valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
    if not valid_indices:
        return None
    return min(valid_indices, key=lambda index: (-rewards[index], index))


def chunk_metadata(
    *,
    episode_id: str,
    decision_id: int,
    selected_chunk: Iterable[int],
    raw_model_outputs: Iterable[str],
    parsed_candidate_chunks: Iterable[Iterable[int]],
    candidate_rewards: Iterable[float],
    selected_index: Optional[int],
) -> List[Dict[str, Any]]:
    """Build an instrumentation-only metadata item for each expanded action."""
    selected = [int(action) for action in selected_chunk]
    parsed = [[int(action) for action in actions] for actions in parsed_candidate_chunks]
    raw = [str(output) for output in raw_model_outputs]
    rewards = [float(reward) for reward in candidate_rewards]
    stop_offsets = [offset for offset, action in enumerate(selected) if action == 0]
    common = {
        "decision_id": int(decision_id),
        "chunk_id": f"{episode_id}:{decision_id}",
        "raw_model_outputs": raw,
        "parsed_candidate_chunks": parsed,
        "candidate_rewards": rewards,
        "selected_candidate_index": selected_index,
        "selected_chunk": selected,
        "any_candidate_contains_stop": any(0 in actions for actions in parsed),
        "selected_chunk_contains_stop": bool(stop_offsets),
        "selected_stop_offset": stop_offsets[0] if stop_offsets else None,
    }
    return [
        {
            **common,
            "chunk_offset": offset,
            "selected_future_stop": any(stop > offset for stop in stop_offsets),
        }
        for offset in range(len(selected))
    ]


def termination_reason(
    *,
    episode_over_after: bool,
    max_step_override: bool,
    early_rotation_override: bool,
    stop_executed: bool,
    action_source: str,
) -> Optional[str]:
    if not episode_over_after:
        return None
    if max_step_override:
        return "max_step_stop"
    if early_rotation_override:
        return "early_rotation_stop"
    if stop_executed and action_source == "generated":
        return "generated_stop"
    if stop_executed and action_source == "pending":
        return "queued_stop"
    if not stop_executed:
        return "environment_done_after_non_stop"
    return "environment_done_other"


class StopTraceWriter:
    """Append validated JSON records without participating in action choice."""

    def __init__(self, path: str | Path, episode_id: str, success_distance: float):
        self.path = Path(path)
        self.episode_id = str(episode_id)
        self.success_distance = float(success_distance)
        self.env_step = 0
        self.first_oracle_reached = False
        self.minimum_distance: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def record(
        self,
        *,
        distance_before: float,
        distance_after: float,
        episode_over_before: bool,
        episode_over_after: bool,
        agent_action: Optional[int],
        action_source: str,
        overridden_action: Optional[int],
        override_reason: Optional[str],
        executed_action: int,
        pending_before: int,
        pending_after: int,
        metadata: Optional[Dict[str, Any]],
        early_rotation_override: bool,
        max_step_override: bool,
    ) -> Dict[str, Any]:
        before = float(distance_before)
        after = float(distance_after)
        within_before = before < self.success_distance
        within_after = after < self.success_distance
        first_reach = not self.first_oracle_reached and (within_before or within_after)
        self.first_oracle_reached = self.first_oracle_reached or within_before or within_after
        distances = [before, after]
        self.minimum_distance = min(
            distances if self.minimum_distance is None else [self.minimum_distance, *distances]
        )
        meta = dict(metadata or {})
        stop_executed = int(executed_action) == 0
        record = {
            "episode_id": self.episode_id,
            "env_step": self.env_step,
            "decision_id": meta.get("decision_id"),
            "chunk_id": meta.get("chunk_id"),
            "chunk_offset": meta.get("chunk_offset"),
            "distance_to_goal_before": before,
            "distance_to_goal_after": after,
            "success_distance": self.success_distance,
            "success_distance_config_source": SUCCESS_DISTANCE_CONFIG_SOURCE,
            "within_success_radius_before": within_before,
            "within_success_radius_after": within_after,
            "first_oracle_reach": first_reach,
            "episode_over_before": bool(episode_over_before),
            "episode_over_after": bool(episode_over_after),
            "agent_action": agent_action,
            "action_source": action_source,
            "overridden_action": overridden_action,
            "override_reason": override_reason,
            "executed_action": int(executed_action),
            "pending_queue_length_before": int(pending_before),
            "pending_queue_length_after": int(pending_after),
            "stop_is_current_queued_action": (
                meta.get("agent_action_source", action_source) == "pending"
                and agent_action == 0
            ),
            "raw_model_outputs": meta.get("raw_model_outputs", []),
            "parsed_candidate_chunks": meta.get("parsed_candidate_chunks", []),
            "candidate_rewards": meta.get("candidate_rewards", []),
            "selected_candidate_index": meta.get("selected_candidate_index"),
            "selected_chunk": meta.get("selected_chunk", []),
            "any_candidate_contains_stop": bool(meta.get("any_candidate_contains_stop", False)),
            "selected_chunk_contains_stop": bool(meta.get("selected_chunk_contains_stop", False)),
            "selected_stop_offset": meta.get("selected_stop_offset"),
            "selected_future_stop": bool(meta.get("selected_future_stop", False)),
            "stop_executed": stop_executed,
            "early_rotation_override": bool(early_rotation_override),
            "max_step_override": bool(max_step_override),
            "environment_ended_after_non_stop": bool(episode_over_after and not stop_executed),
            "done": bool(episode_over_after),
        }
        record["termination_reason"] = termination_reason(
            episode_over_after=bool(episode_over_after),
            max_step_override=bool(max_step_override),
            early_rotation_override=bool(early_rotation_override),
            stop_executed=stop_executed,
            action_source=action_source,
        )
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        self.env_step += 1
        return record

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
