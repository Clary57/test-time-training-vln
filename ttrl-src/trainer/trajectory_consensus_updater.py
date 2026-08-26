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
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
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
            generated_ids = self.model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=True,
            )
        # Tensors created in inference_mode cannot be saved by the subsequent
        # autograd graph.  Clone outside the context before teacher-forcing the
        # sampled tokens through the trainable LoRA model.
        prompt_completion_ids = generated_ids.clone()
        timing["generation"] = time.perf_counter() - generation_start

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = self._completion_mask(completion_ids)
        completions = self.processor.batch_decode(completion_ids, skip_special_tokens=True)
        actions, rewards_list, valid_list = self._actions_and_rewards(completions)
        valid_count = int(sum(valid_list))
        selected_candidate_index = self._selected_candidate_index(rewards_list, valid_list)
        selected_actions = self._actions_for_selected_index(actions, selected_candidate_index)

        loss_forward_start = time.perf_counter()
        self.model.train()
        prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(
            self.num_generations, dim=0
        )
        attention_mask = torch.cat([prompt_mask, completion_mask.to(prompt_mask.dtype)], dim=1)
        per_token_logps = self._per_token_logps(
            self.model,
            prompt_completion_ids,
            attention_mask,
            self._repeat_pixel_values(prompt_inputs["pixel_values"], self.num_generations),
            prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0),
        )[:, prompt_length - 1 :]

        rewards = torch.tensor(rewards_list, dtype=per_token_logps.dtype, device=per_token_logps.device)
        valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=per_token_logps.device)
        advantages = torch.zeros_like(rewards)
        if valid_count:
            valid_rewards = rewards[valid_mask]
            advantages[valid_mask] = (valid_rewards - valid_rewards.mean()) / (
                valid_rewards.std(unbiased=False) + 1e-4
            )
        # The detached copy makes this the usual score-function surrogate while
        # retaining a differentiable tensor for the all-equal-reward case.
        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_candidate_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)
        loss = per_candidate_loss[valid_mask].mean() if valid_count else per_token_logps.sum() * 0.0
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
        if not self.disable_update:
            self.optimizer.step()
        timing["optimizer"] = time.perf_counter() - optimizer_start
        after_norm = self.adapter_norm()
        after_checksum = self.adapter_checksum() if self.disable_update else None
        parameters_unchanged = None
        if self.disable_update:
            parameters_unchanged = all(
                torch.equal(parameter.detach().cpu(), before)
                for (_, parameter), before in zip(self.trainable_parameters, before_parameters)
            )
            if not parameters_unchanged or before_checksum != after_checksum or before_norm != after_norm:
                raise RuntimeError("LoRA parameters changed while optimizer updates were disabled")

        result = {
            "selected_actions": selected_actions,
            "loss": float(loss.detach().item()),
            "rewards": [float(reward) for reward in rewards_list],
            "reward_std": float(np.std([rewards_list[i] for i, valid in enumerate(valid_list) if valid])) if valid_count else 0.0,
            "valid_count": valid_count,
            "timing": timing,
            "adapter_norm_before": before_norm,
            "adapter_norm_after": after_norm,
            "adapter_checksum_before": before_checksum,
            "adapter_checksum_after": after_checksum,
            "parameters_unchanged": parameters_unchanged,
            "update_enabled": not self.disable_update,
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
                distance, _ = fastdtw(
                    trajectories[i], trajectories[j],
                    dist=lambda first, second: np.linalg.norm(np.asarray(first[:2]) - np.asarray(second[:2])),
                )
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
        trajectory = [[x, y, theta]]
        for action in action_sequence:
            if action == 1:
                x += (self.forward_distance / 100.0) * math.cos(theta)
                y += (self.forward_distance / 100.0) * math.sin(theta)
            elif action == 2:
                theta += math.radians(self.turn_angle)
            elif action == 3:
                theta -= math.radians(self.turn_angle)
            theta = (theta + math.pi) % (2 * math.pi) - math.pi
            trajectory.append([x, y, theta])
        return trajectory
