from peft import PeftConfig
import os, transformers
from transformers.trainer import (
Trainer, 
PreTrainedModel,
)
from typing import Optional, Union, Any
import torch
import torch.nn as nn
from packaging import version
from transformers import (
Qwen2_5_VLForConditionalGeneration,
AutoProcessor,
GenerationConfig,
)
from peft import get_peft_model
from src.trainer.episode_state import Episode_State
from src.trainer.path_planner import DiscretePathPlanner
from transformers.utils import (
    is_apex_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    is_accelerate_available
)
from qwen_vl_utils import process_vision_info
from trl.models import unwrap_model_for_generation
from transformers.training_args import OptimizerNames
import re, math, random, json
import numpy as np
from collections import defaultdict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
if is_accelerate_available():
    from accelerate.utils import DistributedType
if is_apex_available():
    from apex import amp


logger = transformers.trainer.logger



class TTTRLTrainer(Trainer):
    def __init__(self, 
        model: Union[str, PreTrainedModel],
        # ref_model: Union[str, PreTrainedModel],
        simulator_config_path: str = None,
        peft_config: Optional[PeftConfig] = None,
        num_generations: int = 8,
        temperature: float = 1.0,
        log_save_path: str = None,
        *args, 
        **kwargs
    ):
        self.num_generations = num_generations

        if isinstance(model, str):
            model_id = model
            model_init_kwargs = {}
            model_init_kwargs["attn_implementation"] = 'flash_attention_2'
            model_init_kwargs['torch_dtype'] = torch.bfloat16
            model_init_kwargs["use_cache"] = (
                    False if kwargs['args'].gradient_checkpointing else True
                )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.processor.image_processor.max_pixels = 501760
        else:
            raise NotImplementedError
        
        self.ref_model = None
        
        ## Freeze the visual encoder
        for p in model.visual.parameters():
            p.requires_grad = False

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            model.enable_input_require_grads()
        
        # trainable_params = [k for k,v in model.named_parameters() if v.requires_grad]
        # print(f"trainable param = {trainable_params}")
        
        self.model = model

        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=512,
            top_p=1.0,
            use_cache=True,
            repetition_penalty = 1.05,
            num_return_sequences = num_generations
        )

        self.beta = 0.05

        self.path_planner = DiscretePathPlanner()

        self._metrics = defaultdict(list)

        self.log_save_path = log_save_path
        os.makedirs(os.path.join(self.log_save_path, "log"), exist_ok=True)

        super().__init__(model = model, *args, **kwargs)

        self.episode_states = [
            Episode_State(
                simulator_config_path,
                local_rank = self.accelerator.local_process_index,
            ) for _ in range(self._train_batch_size)
        ]

        # print(f"********** Rank = {self.accelerator.local_process_index}, train batch size = {self._train_batch_size} **********")
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def sync_done(self):
        
        local_alive_mask = [s.alive for s in self.episode_states]
        local_alive_mask = torch.tensor(
            local_alive_mask,
            device=self.accelerator.device,
        )

        local_done = sum(not s.alive for s in self.episode_states)

        local_done = torch.tensor(
            local_done,
            device=self.accelerator.device,
        )
        global_done = self.accelerator.reduce(
            local_done,
            reduction="sum",
        )
        return global_done.item(), local_alive_mask
    
    
    
    def _prepare_inputs(
        self, 
        inputs: dict[str, Union[torch.Tensor, Any]],
        reset_env: bool = False,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        conversations = []
        if reset_env:
            episode_ids = [x["episode_id"] for x in inputs]
            assert len(episode_ids) == len(self.episode_states), f"The number of episode IDs {len(episode_ids)} must match the number of episode states. {len(self.episode_states)}"

            for state, episode_id in zip(self.episode_states, episode_ids):
                state.reset(episode_id)
                conversation = state.build_prompt()
                conversations.append(conversation)
        else:
            for state in self.episode_states:
                conversation = state.build_prompt()
                conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                for conversation in conversations
        ]
        image_inputs = []
        for conversation in conversations:
            imgs, vids = process_vision_info(conversation)
            imgs = [item.resize((308,252)) for item in imgs]
            image_inputs.append(imgs)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        batch = batch.to(self.accelerator.device)
        return batch
    
    def training_step_per_episode(
        self,
        model: nn.Module, 
        inputs: dict[str, Union[torch.Tensor, Any]], 
        num_items_in_batch=None,
        local_alive_mask=None,
    ):
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch,local_alive_mask=local_alive_mask)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
    

    def training_step(
        self, 
        model: nn.Module, 
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        ## reset the env and then get the collected results
        current_inputs = self._prepare_inputs(inputs, reset_env=True)
        global_done, local_alive_mask = self.sync_done()
        
        while global_done != self.accelerator.num_processes * self.args.train_batch_size:
            
            loss = self.training_step_per_episode(
                model, 
                current_inputs,
                num_items_in_batch=num_items_in_batch,
                local_alive_mask = local_alive_mask,
            )

            global_done, local_alive_mask = self.sync_done()
            current_inputs = self._prepare_inputs(inputs, reset_env=False)

        for state in self.episode_states:
            result_dict = state.get_metrics()
            with open(os.path.join(os.path.join(self.log_save_path, "log"),"stats_{}.json".format(state.episode_id)), "w") as f:
                json.dump(result_dict, f, indent=4)
        
        return loss
    
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs=False, 
        num_items_in_batch=None,
        local_alive_mask=None, # tensor of shape [B,]
    ):
        if return_outputs:
            raise ValueError("The TTTRLTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processor.tokenizer.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]


        # with torch.inference_mode():
        #     if self.ref_model is not None:
        #         ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        #     else:
        #         with self.accelerator.unwrap_model(model).disable_adapters():
        #             ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # # Compute the KL divergence between the model and the reference model
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        # size = (B*G, L)
        completions = self.processor.batch_decode(completion_ids, skip_special_tokens=True)

        
        rewards, completion_valid_mask = self.compute_reward_and_execute_actions(completions)
        rewards = rewards.reshape(-1) # tensors of [B*G]
        completion_valid_mask = completion_valid_mask.reshape(-1) # tensors of [B*G]


        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        per_token_loss = -per_token_loss # [B*G, L]

        loss_valid_mask = completion_valid_mask * local_alive_mask.repeat_interleave(self.num_generations)
        loss_valid_mask = loss_valid_mask.unsqueeze(1)
        per_token_loss = per_token_loss * loss_valid_mask
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()


        log_rewards = self.accelerator.gather_for_metrics(rewards).mean(0)

        self._metrics[f"rewards"].append(log_rewards.item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


    def compute_reward_and_execute_actions(self, completions):
        '''
        completions:  size of (B*G, L)
        return:
            rewards: tensors of [B, G]
        '''
        grouped_completions = [[completions[i] for i in range(j, j + self.num_generations)] for j in range(0, len(completions), self.num_generations)]

        reward_list = []

        # some answer mask may not parsable
        completion_valid_mask = []

        for batch_idx, sample_group in enumerate(grouped_completions):
            current_sample_action_list = []
            completion_valid_mask_this_group = [1 for _ in range(len(sample_group))]
            for rollout_idx, item in enumerate(sample_group):
                current_rollout_action_list = []
                grouped_actions = self.extract_multi_result(item) # [ [action_index, numeric] ]
                select_action_idx = 2
                grouped_actions = grouped_actions[:select_action_idx]

                for idx, (action_index,numeric) in enumerate(grouped_actions):
                    if action_index == 0:
                        current_rollout_action_list.append(0)
                    elif action_index == 1:
                        for _ in range(min(3, round(numeric/25))):
                            current_rollout_action_list.append(1)

                    elif action_index == 2:
                        for _ in range(min(3,round(numeric/15))):
                            current_rollout_action_list.append(2)

                    elif action_index == 3:
                        for _ in range(min(3,round(numeric/15))):
                            current_rollout_action_list.append(3)
                    
                    if action_index is None:
                        print(f'random select an action {action_index}, {numeric}, {item}')
                        action_index = random.randint(1, 3)
                        current_rollout_action_list.append(action_index)
                        completion_valid_mask_this_group[rollout_idx] = 0

                current_sample_action_list.append(current_rollout_action_list)

            rewards_this_sample, baseline_action = self.rewards_function(current_sample_action_list,completion_valid_mask_this_group)
            reward_list.append(rewards_this_sample)
            completion_valid_mask.append(completion_valid_mask_this_group)


            ## Update the corresponding env
            self.episode_states[batch_idx].act(baseline_action)

        reward_list = np.array(reward_list) # faster tensor
        rewards = torch.tensor(
            reward_list, device = self.accelerator.device
        )
        completion_valid_mask = np.array(completion_valid_mask)
        completion_valid_mask = torch.tensor(
            completion_valid_mask, device = self.accelerator.device
        )

        return rewards, completion_valid_mask



    # def rewards_function(self, action_list):
    #     '''
    #     action_list:  batched action_list, size of [G, L]

    #     return:
    #         rewards: list of [G]
    #     '''

    #     # calculate the selected baseline
    #     # baseline in the form of action sequence
    #     pos_list = []
    #     stop_count = 0
    #     for action_seq in action_list:
    #         stop_count = stop_count + 1 if 0 in action_seq else stop_count
    #         pos_list.append(self.get_traj_from_action(action_seq)[-1])
    #     pos_list = np.array(pos_list) # (G, 3)
    #     mean_pos = np.mean(pos_list, axis=0) # (3)

    #     ## Compute selected baseline model
    #     baseline_action = self.path_planner.plan_single_point(mean_pos)
    #     if stop_count > len(action_list) // 2 and baseline_action[-1] != 0:
    #         baseline_action.append(0)

    #     # Calculate rewards
    #     # 

    #     # distance of each node to the selected center
    #     dist = np.linalg.norm(pos_list - mean_pos, axis=1)

    #     max_dist = dist.max() + 1e-6

    #     reward_min = -1.0
    #     reward_max = 1.0

    #     rewards = reward_max - (reward_max - reward_min) * dist / max_dist
    #     rewards = np.clip(rewards, reward_min, reward_max)

    #     return rewards, baseline_action


    def rewards_function(self, action_list, valid_mask):
        '''
        action_list:  batched action_list, size of [G, L]
        valid_mask: batched valid_mask, size of [G]
        return:
            rewards: list of [G]
        '''

        # calculate the selected baseline
        # baseline in the form of action sequence
        G = len(action_list)
        valid_mask = np.asarray(valid_mask, dtype=bool)

        # default all invalid rollout with reward = -1
        rewards = np.full(G, -1.0, dtype=np.float32)

        valid_indices = np.where(valid_mask)[0]

        # no valid
        if len(valid_indices) == 0:
            return rewards, action_list[0]

        # only one valid
        if len(valid_indices) == 1:
            rewards[valid_indices[0]] = 1.0
            return rewards, action_list[valid_indices[0]]

        # -------------------------
        # build trajectories
        # -------------------------
        traj_list = [
            self.get_traj_from_action(action_list[idx])
            for idx in valid_indices
        ]
        stop_count = sum([1 if 0 in action_list[idx] else 0 for idx in valid_indices])

        def euclidean_distance(a, b):
            return np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2]))

        # -------------------------
        # pairwise DTW
        # -------------------------
        V = len(valid_indices)
        dist_matrix = np.zeros((V, V), dtype=np.float32)
        for i in range(G):
            for j in range(i + 1, G):
                d, _ = fastdtw(
                    traj_list[i],
                    traj_list[j],
                    dist=euclidean_distance,
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # -------------------------
        # consensus rollout
        # -------------------------
        mean_dist = dist_matrix.mean(axis=1)
        baseline_local = np.argmin(mean_dist)
        baseline_global = valid_indices[baseline_local]
        baseline_action = action_list[baseline_global]
        # -------------------------
        # reward for valid rollout
        # -------------------------
        d_min = mean_dist.min()
        d_max = mean_dist.max()
        if d_max - d_min < 1e-6:
            valid_rewards = np.ones(V, dtype=np.float32)
        else:
            valid_rewards = 1.0 - (mean_dist - d_min) / (d_max - d_min)
            valid_rewards = valid_rewards * 2.0 - 1.0
        rewards[valid_indices] = valid_rewards
        return rewards, baseline_action

    
        
    

    def get_traj_from_action(self, action_sequence):

        x = 0.0
        y = 0.0
        theta = 0.0
        STEP_DISTANCE = 0.25 
        TURN_ANGLE = math.radians(15)

        pos_list = [[0.0, 0.0, 0.0] for _ in range(len(action_sequence)+1)]
        
        for idx, action in enumerate(action_sequence):
            if action == 1:
                x += STEP_DISTANCE * math.cos(theta)
                y += STEP_DISTANCE * math.sin(theta)
            elif action == 2:
                theta += TURN_ANGLE
            elif action == 3:
                theta -= TURN_ANGLE
            else:
                pass
            theta = (theta + math.pi) % (2 * math.pi) - math.pi
            pos_list[idx+1] = [x, y, theta]
        
        return pos_list

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, 25
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, 15
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, 15
            match = match.group()
            return 3, float(match)
        return None, None 
    
    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result
    

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
    