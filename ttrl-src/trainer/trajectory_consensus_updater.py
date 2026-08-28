"""Single-process, episode-local trajectory-consensus test-time updates.

This module deliberately does not inherit from ``transformers.Trainer`` and does
not use Accelerate.  The evaluator owns one updater and calls ``reset_episode``
before an episode, followed by ``adapt_step`` once per newly generated action
chunk.  Pending atomic actions are executed without another call to this module.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastdtw import fastdtw
from transformers import GenerationConfig


class TrajectoryConsensusUpdater:
    """Update LoRA parameters once after every generated navigation chunk."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        *,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        num_generations: int,
        temperature: float,
        seed: int,
        disable_update: bool = False,
        trace_metadata_enabled: bool = False,
        sequential_generation: bool = False,
        update_policy: str = "legacy",
        min_reward_std: float = 0.2,
        min_reward_margin: float = 0.25,
        action_span_only: bool = False,
        max_updates_per_episode: int = 0,
        anchor_l2_weight: float = 0.0,
        max_adapter_delta_norm: float = 0.0,
        pairwise_beta: float = 1.0,
        reward_geometry: str = "xy",
        orientation_distance_scale: float = 0.25 / math.radians(15.0),
        stop_distance_penalty: float = 0.25,
        candidate_sampling: str = "all_stochastic",
        forward_distance: float = 25.0,
        turn_angle: float = 15.0,
    ) -> None:
        if num_generations < 1:
            raise ValueError("num_generations must be at least one")
        self.model = model
        self.processor = processor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_generations = num_generations
        self.seed = seed
        self.disable_update = disable_update
        self.trace_metadata_enabled = trace_metadata_enabled
        self.sequential_generation = sequential_generation
        if update_policy not in {"legacy", "gated"}:
            raise ValueError("update_policy must be 'legacy' or 'gated'")
        if min_reward_std < 0 or min_reward_margin < 0:
            raise ValueError("reward gate thresholds must be non-negative")
        if max_updates_per_episode < 0:
            raise ValueError("max_updates_per_episode must be non-negative")
        if anchor_l2_weight < 0 or max_adapter_delta_norm < 0:
            raise ValueError("adapter drift controls must be non-negative")
        if pairwise_beta <= 0:
            raise ValueError("pairwise_beta must be positive")
        if reward_geometry not in {"xy", "se2_stop"}:
            raise ValueError("reward_geometry must be 'xy' or 'se2_stop'")
        if orientation_distance_scale < 0 or stop_distance_penalty < 0:
            raise ValueError("trajectory distance scales must be non-negative")
        if candidate_sampling not in {"all_stochastic", "greedy_plus_stochastic"}:
            raise ValueError(
                "candidate_sampling must be 'all_stochastic' or 'greedy_plus_stochastic'"
            )
        self.update_policy = update_policy
        self.min_reward_std = min_reward_std
        self.min_reward_margin = min_reward_margin
        self.action_span_only = action_span_only
        self.max_updates_per_episode = max_updates_per_episode
        self.anchor_l2_weight = anchor_l2_weight
        self.max_adapter_delta_norm = max_adapter_delta_norm
        self.pairwise_beta = pairwise_beta
        self.reward_geometry = reward_geometry
        self.orientation_distance_scale = orientation_distance_scale
        self.stop_distance_penalty = stop_distance_penalty
        self.candidate_sampling = candidate_sampling
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=512,
            top_p=1.0,
            use_cache=True,
            repetition_penalty=1.05,
            num_return_sequences=num_generations,
        )
        self.trainable_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not self.trainable_parameters:
            raise ValueError("TrajectoryConsensusUpdater requires trainable LoRA parameters")
        self.base_adapter_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.trainable_parameters
        }
        self.optimizer = self._new_optimizer()
        self.update_opportunity_count = 0
        self.optimizer_step_count = 0

    def _candidate_generation_configs(self) -> List[GenerationConfig]:
        """Return generation configs in stable candidate order."""
        if self.candidate_sampling == "all_stochastic":
            if not self.sequential_generation:
                return [self.generation_config]
            configs = []
            for _ in range(self.num_generations):
                config = GenerationConfig.from_dict(self.generation_config.to_dict())
                config.num_return_sequences = 1
                configs.append(config)
            return configs

        greedy = GenerationConfig.from_dict(self.generation_config.to_dict())
        greedy.do_sample = False
        greedy.num_return_sequences = 1
        if self.num_generations == 1:
            return [greedy]
        stochastic = []
        for _ in range(self.num_generations - 1):
            config = GenerationConfig.from_dict(self.generation_config.to_dict())
            config.num_return_sequences = 1
            stochastic.append(config)
        return [greedy, *stochastic]

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [parameter for _, parameter in self.trainable_parameters],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def _episode_seed(seed: int, episode_id: str) -> int:
        digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31)

    def reset_episode(self, episode_id: str | int) -> None:
        """Restore the initial LoRA tensors and discard all optimizer momentum."""
        self.episode_id = str(episode_id)
        self.episode_seed = self._episode_seed(self.seed, self.episode_id)
        random.seed(self.episode_seed)
        np.random.seed(self.episode_seed)
        torch.manual_seed(self.episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.episode_seed)

        with torch.no_grad():
            for name, parameter in self.trainable_parameters:
                parameter.copy_(self.base_adapter_state[name].to(parameter.device, parameter.dtype))
        self.optimizer = self._new_optimizer()
        self.update_opportunity_count = 0
        self.optimizer_step_count = 0

    def adapter_norm(self) -> float:
        with torch.no_grad():
            squared_norm = sum(
                parameter.detach().float().pow(2).sum()
                for _, parameter in self.trainable_parameters
            )
            return float(squared_norm.sqrt().item())

    def adapter_checksum(self) -> str:
        """Return a deterministic byte-level checksum of all trainable adapters."""
        digest = hashlib.sha256()
        for name, parameter in self.trainable_parameters:
            tensor = parameter.detach().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    def adapter_delta_norm(self) -> float:
        """Return the L2 distance from the episode's initial adapter."""
        with torch.no_grad():
            squared_norm = sum(
                (parameter.detach().float() - self.base_adapter_state[name].to(
                    parameter.device, torch.float32
                )).pow(2).sum()
                for name, parameter in self.trainable_parameters
            )
            return float(squared_norm.sqrt().item())

    def _anchor_l2_loss(self) -> torch.Tensor:
        squared_sum = None
        element_count = 0
        for name, parameter in self.trainable_parameters:
            difference = parameter.float() - self.base_adapter_state[name].to(
                parameter.device, torch.float32
            )
            term = difference.pow(2).sum()
            squared_sum = term if squared_sum is None else squared_sum + term
            element_count += parameter.numel()
        return squared_sum / max(1, element_count)

    @staticmethod
    def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
        if not subsequence or len(subsequence) > len(sequence):
            return None
        for start in range(len(sequence) - len(subsequence) + 1):
            if list(sequence[start:start + len(subsequence)]) == list(subsequence):
                return start
        return None

    def _action_span_mask(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completion: str,
    ) -> Tuple[torch.Tensor, bool]:
        """Mask only answer/action tokens without changing generated token IDs.

        Plain action-only generations already consist entirely of action tokens.
        Tagged completions must expose an exact tokenized answer subsequence; if
        they do not, gated mode skips the update rather than training unrelated
        explanation or formatting tokens.
        """
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if answer_match is None:
            # Current NaVIDA generations are plain action strings.  In that
            # case the whole textual completion is the action span, but EOS is
            # a sequence-control token rather than part of the navigation action.
            mask = completion_mask.clone()
            mask[completion_ids == self.processor.tokenizer.eos_token_id] = 0
            return mask, bool(mask.sum().item())
        answer = answer_match.group(1).strip()
        if not answer:
            return torch.zeros_like(completion_mask), False
        token_ids = completion_ids.detach().cpu().tolist()
        valid_length = int(completion_mask.sum().item())
        token_ids = token_ids[:valid_length]
        for candidate_text in (answer, f" {answer}"):
            encoded = self.processor.tokenizer.encode(
                candidate_text, add_special_tokens=False
            )
            start = self._find_subsequence(token_ids, encoded)
            if start is None:
                continue
            mask = torch.zeros_like(completion_mask)
            mask[start:start + len(encoded)] = 1
            return mask, True
        return torch.zeros_like(completion_mask), False

    def _action_span_masks(
        self,
        completion_id_batches: Sequence[torch.Tensor],
        completion_masks: Sequence[torch.Tensor],
        completions: Sequence[str],
    ) -> Tuple[List[torch.Tensor], List[bool]]:
        masks: List[torch.Tensor] = []
        available: List[bool] = []
        completion_index = 0
        for ids_batch, mask_batch in zip(completion_id_batches, completion_masks):
            batch_masks = []
            for ids, base_mask in zip(ids_batch, mask_batch):
                span_mask, found = self._action_span_mask(
                    ids, base_mask, completions[completion_index]
                )
                batch_masks.append(span_mask)
                available.append(found)
                completion_index += 1
            masks.append(torch.stack(batch_masks))
        return masks, available

    def _update_gate(
        self,
        rewards: Sequence[float],
        valid: Sequence[bool],
        action_spans_available: Sequence[bool],
    ) -> Tuple[bool, str, float, float]:
        valid_rewards = [float(reward) for reward, is_valid in zip(rewards, valid) if is_valid]
        reward_std = float(np.std(valid_rewards)) if valid_rewards else 0.0
        sorted_rewards = sorted(valid_rewards, reverse=True)
        reward_margin = (
            sorted_rewards[0] - sorted_rewards[1] if len(sorted_rewards) >= 2 else 0.0
        )
        if self.update_policy == "legacy":
            return True, "legacy_policy", reward_std, reward_margin
        if len(valid_rewards) != self.num_generations:
            return False, "not_all_candidates_valid", reward_std, reward_margin
        if reward_std < self.min_reward_std:
            return False, "reward_std_below_threshold", reward_std, reward_margin
        best_count = sum(abs(reward - sorted_rewards[0]) < 1e-8 for reward in valid_rewards)
        if best_count != 1:
            return False, "best_candidate_not_unique", reward_std, reward_margin
        if reward_margin < self.min_reward_margin:
            return False, "reward_margin_below_threshold", reward_std, reward_margin
        if self.action_span_only and not all(action_spans_available):
            return False, "action_span_unavailable", reward_std, reward_margin
        if (
            self.max_updates_per_episode > 0
            and self.update_opportunity_count >= self.max_updates_per_episode
        ):
            return False, "episode_update_limit_reached", reward_std, reward_margin
        return True, "eligible", reward_std, reward_margin

    def _parameter_snapshot(self) -> List[torch.Tensor]:
        return [parameter.detach().cpu().clone() for _, parameter in self.trainable_parameters]

    def _restore_parameter_snapshot(self, snapshot: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            for (_, parameter), before in zip(self.trainable_parameters, snapshot):
                parameter.copy_(before.to(parameter.device, parameter.dtype))

    @classmethod
    def _clone_optimizer_value_to_cpu(cls, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cls._clone_optimizer_value_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._clone_optimizer_value_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_optimizer_value_to_cpu(item) for item in value)
        return value

    def _optimizer_snapshot(self) -> Dict[str, Any]:
        return self._clone_optimizer_value_to_cpu(self.optimizer.state_dict())

    def _restore_optimizer_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.optimizer = self._new_optimizer()
        self.optimizer.load_state_dict(snapshot)

    def _completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.processor.tokenizer.eos_token_id
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full(
            (completion_ids.size(0),),
            completion_ids.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        positions = torch.arange(completion_ids.size(1), device=completion_ids.device)
        return (positions.unsqueeze(0) <= eos_idx.unsqueeze(1)).to(torch.float32)

    @staticmethod
    def _repeat_pixel_values(pixel_values: torch.Tensor, count: int) -> torch.Tensor:
        return pixel_values.repeat((count,) + (1,) * (pixel_values.ndim - 1))

    @staticmethod
    def _per_token_logps(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        ).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_logps = []
        for logits_row, labels_row in zip(logits, labels):
            token_logps.append(
                torch.gather(
                    logits_row.log_softmax(dim=-1), 1, labels_row.unsqueeze(1)
                ).squeeze(1)
            )
        return torch.stack(token_logps)

    def adapt_step(self, prompt_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate candidates, update LoRA once, and return the pre-update medoid."""
        if prompt_inputs["input_ids"].size(0) != 1:
            raise ValueError("Online TTRL evaluates exactly one episode/prompt at a time")
        timing: Dict[str, float] = {}
        before_norm = self.adapter_norm()
        before_checksum = self.adapter_checksum() if self.disable_update else None
        before_parameters = (
            # Keep the verification snapshot off-device. Production runs are
            # close to GPU capacity, so validation must not duplicate LoRA
            # tensors in GPU memory.
            [parameter.detach().cpu().clone() for _, parameter in self.trainable_parameters]
            if self.disable_update else None
        )

        generation_start = time.perf_counter()
        self.model.eval()
        with torch.inference_mode():
            candidate_generation_configs = self._candidate_generation_configs()
            generated_batches = [self.model.generate(
                    **prompt_inputs,
                    generation_config=generation_config,
                    use_model_defaults=True,
                ) for generation_config in candidate_generation_configs]
        separate_candidate_batches = len(generated_batches) > 1
        # Tensors created in inference_mode cannot be saved by the subsequent
        # autograd graph.  Clone outside the context before teacher-forcing the
        # sampled tokens through the trainable LoRA model.
        prompt_completion_batches = [batch.clone() for batch in generated_batches]
        timing["generation"] = time.perf_counter() - generation_start

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_id_batches = [batch[:, prompt_length:] for batch in prompt_completion_batches]
        completion_masks = [self._completion_mask(batch) for batch in completion_id_batches]
        completions = [
            text
            for batch in completion_id_batches
            for text in self.processor.batch_decode(batch, skip_special_tokens=True)
        ]
        actions, rewards_list, valid_list = self._actions_and_rewards(completions)
        valid_count = int(sum(valid_list))
        selected_candidate_index = self._selected_candidate_index(rewards_list, valid_list)
        selected_actions = self._actions_for_selected_index(actions, selected_candidate_index)
        action_span_masks, action_spans_available = self._action_span_masks(
            completion_id_batches, completion_masks, completions
        ) if self.action_span_only else (completion_masks, [True] * len(completions))
        update_eligible, update_gate_reason, reward_std, reward_margin = self._update_gate(
            rewards_list, valid_list, action_spans_available
        )
        if self.update_policy == "gated" and update_eligible and before_checksum is None:
            before_checksum = self.adapter_checksum()

        loss_forward_start = time.perf_counter()
        self.model.train()
        if separate_candidate_batches:
            per_token_logps_batches = []
            for prompt_completion_ids, completion_mask in zip(
                prompt_completion_batches, completion_masks
            ):
                prompt_mask = prompt_inputs["attention_mask"]
                attention_mask = torch.cat(
                    [prompt_mask, completion_mask.to(prompt_mask.dtype)], dim=1
                )
                per_token_logps_batches.append(self._per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    prompt_inputs["pixel_values"],
                    prompt_inputs["image_grid_thw"],
                )[:, prompt_length - 1 :])
        else:
            completion_mask = completion_masks[0]
            prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(
                self.num_generations, dim=0
            )
            attention_mask = torch.cat([prompt_mask, completion_mask.to(prompt_mask.dtype)], dim=1)
            per_token_logps_batches = [self._per_token_logps(
                self.model,
                prompt_completion_batches[0],
                attention_mask,
                self._repeat_pixel_values(prompt_inputs["pixel_values"], self.num_generations),
                prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0),
            )[:, prompt_length - 1 :]]

        reference_logps = per_token_logps_batches[0]
        rewards = torch.tensor(rewards_list, dtype=reference_logps.dtype, device=reference_logps.device)
        valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=reference_logps.device)
        advantages = torch.zeros_like(rewards)
        if valid_count:
            valid_rewards = rewards[valid_mask]
            advantages[valid_mask] = (valid_rewards - valid_rewards.mean()) / (
                valid_rewards.std(unbiased=False) + 1e-4
            )
        # The detached copy makes this the usual score-function surrogate while
        # retaining a differentiable tensor for the all-equal-reward case.
        if separate_candidate_batches:
            per_candidate_loss = []
            for index, (per_token_logps, completion_mask) in enumerate(
                zip(per_token_logps_batches, action_span_masks)
            ):
                per_token_loss = -torch.exp(
                    per_token_logps - per_token_logps.detach()
                ) * advantages[index]
                per_candidate_loss.append(
                    (per_token_loss * completion_mask).sum()
                    / completion_mask.sum().clamp_min(1)
                )
            per_candidate_loss = torch.stack(per_candidate_loss)
            zero_loss = sum(batch.sum() * 0.0 for batch in per_token_logps_batches)
        else:
            per_token_logps = per_token_logps_batches[0]
            completion_mask = action_span_masks[0]
            per_token_loss = -torch.exp(
                per_token_logps - per_token_logps.detach()
            ) * advantages.unsqueeze(1)
            per_candidate_loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1).clamp_min(1)
            )
            zero_loss = per_token_logps.sum() * 0.0
        legacy_policy_loss = per_candidate_loss[valid_mask].mean() if valid_count else zero_loss
        if self.update_policy == "gated" and update_eligible:
            candidate_mean_logps = []
            for per_token_logps, loss_mask in zip(per_token_logps_batches, action_span_masks):
                candidate_mean_logps.extend(
                    (per_token_logps * loss_mask).sum(dim=1)
                    / loss_mask.sum(dim=1).clamp_min(1)
                )
            candidate_mean_logps = torch.stack(candidate_mean_logps)
            valid_indices = [index for index, is_valid in enumerate(valid_list) if is_valid]
            best_index = selected_candidate_index
            worst_index = min(valid_indices, key=lambda index: (rewards_list[index], index))
            policy_loss = -F.logsigmoid(
                self.pairwise_beta
                * (candidate_mean_logps[best_index] - candidate_mean_logps[worst_index])
            )
            policy_loss_type = "best_vs_worst_action_logp"
        elif self.update_policy == "gated":
            policy_loss = zero_loss
            policy_loss_type = "skipped_by_gate"
        else:
            policy_loss = legacy_policy_loss
            policy_loss_type = "legacy_standardized_advantage"
        anchor_loss = (
            self._anchor_l2_loss()
            if update_eligible and self.anchor_l2_weight > 0
            else zero_loss
        )
        loss = policy_loss + self.anchor_l2_weight * anchor_loss
        timing["loss_forward"] = time.perf_counter() - loss_forward_start

        backward_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        timing["backward"] = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in self.trainable_parameters], self.max_grad_norm
            )
        optimizer_step_executed = False
        update_rolled_back = False
        rollback_reason = None
        parameter_snapshot = None
        optimizer_snapshot = None
        protection_enabled = (
            self.update_policy == "gated" or self.max_adapter_delta_norm > 0
        )
        if update_eligible:
            self.update_opportunity_count += 1
        if not self.disable_update and update_eligible:
            if protection_enabled:
                parameter_snapshot = self._parameter_snapshot()
                optimizer_snapshot = self._optimizer_snapshot()
            self.optimizer.step()
            optimizer_step_executed = True
            if protection_enabled:
                parameters_finite = all(
                    bool(torch.isfinite(parameter.detach()).all().item())
                    for _, parameter in self.trainable_parameters
                )
                adapter_delta_norm = self.adapter_delta_norm()
                if not parameters_finite:
                    rollback_reason = "nonfinite_adapter_parameter"
                elif (
                    self.max_adapter_delta_norm > 0
                    and adapter_delta_norm > self.max_adapter_delta_norm
                ):
                    rollback_reason = "adapter_delta_norm_exceeded"
                if rollback_reason is not None:
                    self._restore_parameter_snapshot(parameter_snapshot)
                    self._restore_optimizer_snapshot(optimizer_snapshot)
                    optimizer_step_executed = False
                    update_rolled_back = True
            if optimizer_step_executed:
                self.optimizer_step_count += 1
        timing["optimizer"] = time.perf_counter() - optimizer_start
        after_norm = self.adapter_norm()
        adapter_delta_norm = self.adapter_delta_norm() if protection_enabled else None
        checksum_required = self.disable_update or (
            self.update_policy == "gated" and update_eligible
        )
        after_checksum = self.adapter_checksum() if checksum_required else None
        parameters_unchanged = None
        if self.disable_update:
            parameters_unchanged = all(
                torch.equal(parameter.detach().cpu(), before)
                for (_, parameter), before in zip(self.trainable_parameters, before_parameters)
            )
            if not parameters_unchanged or before_checksum != after_checksum or before_norm != after_norm:
                raise RuntimeError("LoRA parameters changed while optimizer updates were disabled")
        elif self.update_policy == "gated":
            parameters_unchanged = (
                True if not update_eligible else before_checksum == after_checksum
            )

        result = {
            "selected_actions": selected_actions,
            "loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "policy_loss_type": policy_loss_type,
            "anchor_loss": float(anchor_loss.detach().item()),
            "rewards": [float(reward) for reward in rewards_list],
            "reward_std": reward_std,
            "reward_margin": reward_margin,
            "valid_count": valid_count,
            "timing": timing,
            "adapter_norm_before": before_norm,
            "adapter_norm_after": after_norm,
            "adapter_checksum_before": before_checksum,
            "adapter_checksum_after": after_checksum,
            "parameters_unchanged": parameters_unchanged,
            "update_enabled": optimizer_step_executed,
            "update_eligible": update_eligible,
            "update_gate_reason": update_gate_reason,
            "optimizer_step_executed": optimizer_step_executed,
            "effective_parameter_change": (
                None if parameters_unchanged is None else not parameters_unchanged
            ),
            "update_rolled_back": update_rolled_back,
            "rollback_reason": rollback_reason,
            "episode_update_opportunities": self.update_opportunity_count,
            "episode_optimizer_steps": self.optimizer_step_count,
            "adapter_delta_norm": adapter_delta_norm,
            "action_span_only": self.action_span_only,
            "action_spans_available": action_spans_available,
            "pairwise_beta": self.pairwise_beta,
            "reward_geometry": self.reward_geometry,
            "orientation_distance_scale": self.orientation_distance_scale,
            "stop_distance_penalty": self.stop_distance_penalty,
            "candidate_sampling": self.candidate_sampling,
            "candidate_sampling_modes": [
                "stochastic" if config.do_sample else "greedy"
                for config in candidate_generation_configs
                for _ in range(config.num_return_sequences)
            ],
            "completions": completions,
        }
        if self.trace_metadata_enabled:
            result.update({
                "candidate_actions": [list(action) for action in actions],
                "selected_candidate_index": selected_candidate_index,
            })
        return result

    def _actions_and_rewards(self, completions: Sequence[str]) -> Tuple[List[List[int]], List[float], List[bool]]:
        actions: List[List[int]] = []
        valid: List[bool] = []
        for completion in completions:
            action_sequence, is_valid = self._parse_action_chunk(completion)
            actions.append(action_sequence)
            valid.append(is_valid)
        rewards = self._consensus_rewards(actions, valid)
        return actions, rewards, valid

    def _select_actions(self, actions: Sequence[List[int]], rewards: Sequence[float], valid: Sequence[bool]) -> List[int]:
        selected_index = self._selected_candidate_index(rewards, valid)
        return self._actions_for_selected_index(actions, selected_index)

    @staticmethod
    def _actions_for_selected_index(
        actions: Sequence[List[int]], selected_index: int | None
    ) -> List[int]:
        if selected_index is None:
            # Invalid generations must never receive a gradient.  A deterministic
            # fallback still lets the environment make progress.
            return [random.randint(1, 3)]
        return list(actions[selected_index])

    @staticmethod
    def _selected_candidate_index(
        rewards: Sequence[float], valid: Sequence[bool]
    ) -> int | None:
        valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
        if not valid_indices:
            return None
        return min(valid_indices, key=lambda index: (-rewards[index], index))

    def _parse_action_chunk(self, completion: str) -> Tuple[List[int], bool]:
        actions: List[int] = []
        chunks = [chunk.strip() for chunk in completion.split(",") if chunk.strip()][:2]
        if not chunks:
            return [], False
        for chunk in chunks:
            action, value = self._extract_action(chunk)
            if action is None:
                return [], False
            if action == 0:
                actions.append(0)
            else:
                unit = self.forward_distance if action == 1 else self.turn_angle
                actions.extend([action] * min(3, max(1, round(value / unit))))
        return (actions, bool(actions))

    def _extract_action(self, text: str) -> Tuple[int | None, float | None]:
        answer_match = __import__("re").search(r"<answer>(.*?)</answer>", text, flags=__import__("re").DOTALL)
        text = answer_match.group(1).strip() if answer_match else text.strip()
        lower = text.lower()
        if "stop" in lower:
            return 0, None
        number = __import__("re").search(r"-?\d+(?:\.\d+)?", lower)
        if "forward" in lower:
            return 1, float(number.group()) if number else self.forward_distance
        if "left" in lower:
            return 2, float(number.group()) if number else self.turn_angle
        if "right" in lower:
            return 3, float(number.group()) if number else self.turn_angle
        return None, None

    def _consensus_rewards(self, actions: Sequence[List[int]], valid: Sequence[bool]) -> List[float]:
        rewards = np.full(len(actions), -1.0, dtype=np.float32)
        valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
        if not valid_indices:
            return rewards.tolist()
        if len(valid_indices) == 1:
            rewards[valid_indices[0]] = 1.0
            return rewards.tolist()
        trajectories = [self._trajectory(actions[index]) for index in valid_indices]
        valid_count = len(trajectories)
        distances = np.zeros((valid_count, valid_count), dtype=np.float32)
        for i in range(valid_count):
            for j in range(i + 1, valid_count):
                distance, path = fastdtw(
                    trajectories[i], trajectories[j], dist=self._trajectory_point_distance
                )
                if self.reward_geometry == "se2_stop":
                    distance /= max(1, len(path))
                distances[i, j] = distance
                distances[j, i] = distance
        mean_distance = distances.mean(axis=1)
        if np.ptp(mean_distance) < 1e-6:
            valid_rewards = np.ones(valid_count, dtype=np.float32)
        else:
            valid_rewards = 1.0 - 2.0 * (mean_distance - mean_distance.min()) / np.ptp(mean_distance)
        rewards[valid_indices] = valid_rewards
        return rewards.tolist()

    def _trajectory(self, action_sequence: Sequence[int]) -> List[List[float]]:
        x = y = theta = 0.0
        stopped = 0.0
        trajectory = [
            [x, y, theta, stopped]
            if self.reward_geometry == "se2_stop" else [x, y, theta]
        ]
        for action in action_sequence:
            if action == 0 and self.reward_geometry == "se2_stop":
                stopped = 1.0
            elif action == 1:
                x += (self.forward_distance / 100.0) * math.cos(theta)
                y += (self.forward_distance / 100.0) * math.sin(theta)
            elif action == 2:
                theta += math.radians(self.turn_angle)
            elif action == 3:
                theta -= math.radians(self.turn_angle)
            theta = (theta + math.pi) % (2 * math.pi) - math.pi
            trajectory.append(
                [x, y, theta, stopped]
                if self.reward_geometry == "se2_stop" else [x, y, theta]
            )
            if stopped:
                break
        return trajectory

    def _trajectory_point_distance(self, first: Sequence[float], second: Sequence[float]) -> float:
        if self.reward_geometry == "xy":
            return float(np.linalg.norm(np.asarray(first[:2]) - np.asarray(second[:2])))
        angle_delta = math.atan2(
            math.sin(first[2] - second[2]), math.cos(first[2] - second[2])
        )
        values = (
            first[0] - second[0],
            first[1] - second[1],
            self.orientation_distance_scale * angle_delta,
            self.stop_distance_penalty * (first[3] - second[3]),
        )
        return math.sqrt(sum(value * value for value in values))
