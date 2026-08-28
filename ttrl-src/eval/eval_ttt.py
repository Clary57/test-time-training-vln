import base64
import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os, io
import re
import sys
import torch
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
import argparse, habitat
from episode_subset import load_episode_ids_file, select_episodes_in_file_order, shard_episodes
from habitat_extensions import measures, task
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image
from qwen_vl_utils import process_vision_info
from collections import Counter
from peft import LoraConfig, PeftModel, get_peft_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from trainer.trajectory_consensus_updater import TrajectoryConsensusUpdater
from eval.stop_trace import StopTraceWriter, chunk_metadata, resolve_success_distance

SYSTEM_PROMPT = "You are a helpful assistant."


def navida_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=".*visual.*",
    )


def attach_trainable_lora(model, lora_path, *, native_adapter):
    """Attach LoRA without changing the Qwen model type in official mode."""
    if native_adapter:
        original_type = type(model)
        if lora_path:
            model.load_adapter(lora_path, is_trainable=True)
        else:
            model.add_adapter(navida_lora_config())
        if type(model) is not original_type:
            raise RuntimeError("native LoRA injection unexpectedly changed the model type")
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any("lora_" not in name for name in trainable):
            raise RuntimeError("native LoRA injection exposed non-LoRA trainable parameters")
        return model
    if lora_path:
        return PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    return get_peft_model(model, navida_lora_config())


def encode_image_base64(image):
    """Encode a PIL image for Qwen's image_url conversation format.

    Keeping this local avoids importing vLLM only for a small utility function;
    vLLM and the pinned Transformers version need not be import-compatible for
    Hugging Face evaluation.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def action_id_to_str(action_id):
    # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
    if action_id == 0:
        return "stop"
    elif action_id == 1:
        return f"forward 25 cm"
    elif action_id == 2:
        return f"turn left 15 degree"
    elif action_id == 3:
        return f"turn right 15 degree"
    else:
        raise ValueError(f"Invalid action ID: {action_id}")


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_episode_manifest(result_path, split_num, split_id, episodes, expected_episode_ids,
                           episode_ids_file_sha256=None, selection_seed=None,
                           dataset_split=None, protocol_parameters=None):
    """Persist this shard and the common expected ID set before resume filtering."""
    os.makedirs(result_path, exist_ok=True)
    manifest = {
        "protocol": "episode-local-step-wise-ttrl",
        "split_num": split_num,
        "split_id": split_id,
        "episode_ids": [str(episode.episode_id) for episode in episodes],
        "expected_episode_ids": expected_episode_ids,
        "episode_ids_file_sha256": episode_ids_file_sha256,
        "selection_seed": selection_seed,
        "dataset_split": dataset_split,
        "protocol_parameters": protocol_parameters or {},
    }
    with open(os.path.join(result_path, "episode_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def normalize_episode_ids(dataset):
    """Match episode IDs to the string keys produced by JSON ground-truth files."""
    for episode in dataset.episodes:
        episode.episode_id = str(episode.episode_id)


def evaluate_agent(config, split_id, dataset, model_path, lora_path, result_path, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio,
                    learning_rate, weight_decay, max_grad_norm, temperature, seed,
                    max_episodes=None, early_stop_rotation=25, early_stop_steps=400,
                    disable_update=False, stop_trace=False,
                    official_navida_protocol=False, update_policy="legacy",
                    min_reward_std=0.2, min_reward_margin=0.25,
                    action_span_only=False, max_updates_per_episode=0,
                    anchor_l2_weight=0.0, max_adapter_delta_norm=0.0,
                    pairwise_beta=1.0) -> None:
    log_path = os.path.join(result_path, "log")
    os.makedirs(log_path, exist_ok=True)
    dataset.episodes = [
        episode for episode in dataset.episodes
        if not os.path.exists(os.path.join(log_path, f"stats_{episode.episode_id}.json"))
    ]
    if not dataset.episodes:
        print("All episodes in this split already have results.")
        return

    env = Env(config.habitat, dataset)

    agent = NaVIDA_Agent(model_path, 
                        lora_path, 
                        result_path, 
                        forward_distance, 
                        turn_angle, 
                        max_action_history,
                        resolution_ratio,
                        num_generations,
                        learning_rate,
                        weight_decay,
                        max_grad_norm,
                        temperature,
                        seed,
                        disable_update=disable_update,
                        stop_trace_enabled=stop_trace,
                        official_navida_protocol=official_navida_protocol,
                        update_policy=update_policy,
                        min_reward_std=min_reward_std,
                        min_reward_margin=min_reward_margin,
                        action_span_only=action_span_only,
                        max_updates_per_episode=max_updates_per_episode,
                        anchor_l2_weight=anchor_l2_weight,
                        max_adapter_delta_norm=max_adapter_delta_norm,
                        pairwise_beta=pairwise_beta)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = early_stop_rotation
    EARLY_STOP_STEPS = early_stop_steps

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success", "ndtw"}
    success_distance = resolve_success_distance(config) if stop_trace else None

    count = 0
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    for _ in trange(num_episodes):
        obs = env.reset()
        iter_step = 0
        agent.reset_episode(env.current_episode.episode_id)
        trace_writer = None
        if stop_trace:
            trace_writer = StopTraceWriter(
                os.path.join(log_path, f"stop_trace_{env.current_episode.episode_id}.jsonl"),
                env.current_episode.episode_id,
                success_distance,
            )

        continuse_rotation_count = 0
        last_dtg = 999
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            early_rotation_override = continuse_rotation_count > EARLY_STOP_ROTATION
            max_step_override = iter_step > EARLY_STOP_STEPS
            if early_rotation_override or max_step_override:
                action = {"action": 0}
                action_trace = {
                    "agent_action": None,
                    "action_source": "max_step_override" if max_step_override else "early_rotation_override",
                    "pending_before": len(agent.pending_action_list),
                    "pending_after": len(agent.pending_action_list),
                    "metadata": None,
                }
            else:
                action = agent.act(obs, info, env.current_episode.episode_id)
                action_trace = agent.consume_action_trace_metadata() if stop_trace else None

            iter_step+=1
            obs = env.step(action)
            if trace_writer is not None:
                after_info = env.get_metrics()
                trace_writer.record(
                    distance_before=info["distance_to_goal"],
                    distance_after=after_info["distance_to_goal"],
                    episode_over_before=False,
                    episode_over_after=env.episode_over,
                    agent_action=action_trace["agent_action"],
                    action_source=action_trace["action_source"],
                    overridden_action=None,
                    override_reason=(
                        "max_step" if max_step_override else
                        "early_rotation" if early_rotation_override else None
                    ),
                    executed_action=action["action"],
                    pending_before=action_trace["pending_before"],
                    pending_after=action_trace["pending_after"],
                    metadata=action_trace["metadata"],
                    early_rotation_override=early_rotation_override,
                    max_step_override=max_step_override,
                )
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1

        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)
        agent.finish_episode()
        if trace_writer is not None:
            trace_writer.close()

class NaVIDA_Agent(Agent):
    def __init__(self, model_path, lora_path, result_path, forward_distance,
                    turn_angle, max_action_history, resolution_ratio, num_generations,
                    learning_rate, weight_decay, max_grad_norm, temperature, seed,
                    require_map=True, disable_update=False, stop_trace_enabled=False,
                    official_navida_protocol=False, update_policy="legacy",
                    min_reward_std=0.2, min_reward_margin=0.25,
                    action_span_only=False, max_updates_per_episode=0,
                    anchor_l2_weight=0.0, max_adapter_delta_norm=0.0,
                    pairwise_beta=1.0):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.require_map = require_map
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        self.seed = seed
        self.stop_trace_enabled = stop_trace_enabled
        self.official_navida_protocol = official_navida_protocol
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["use_cache"] = True
        model_init_kwargs['torch_dtype'] = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)

        if lora_path:
            print('Loading trainable LoRA weights...')
        else:
            print('Creating a new trainable LoRA adapter...')
        self.model = attach_trainable_lora(
            self.model,
            lora_path,
            native_adapter=official_navida_protocol,
        )

        self.device = 'cuda'
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_pixels = 501760
        self.updater = TrajectoryConsensusUpdater(
            self.model,
            self.processor,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            num_generations=num_generations,
            temperature=temperature,
            seed=seed,
            disable_update=disable_update,
            trace_metadata_enabled=stop_trace_enabled,
            sequential_generation=official_navida_protocol,
            update_policy=update_policy,
            min_reward_std=min_reward_std,
            min_reward_margin=min_reward_margin,
            action_span_only=action_span_only,
            max_updates_per_episode=max_updates_per_episode,
            anchor_l2_weight=anchor_l2_weight,
            max_adapter_delta_norm=max_adapter_delta_norm,
            pairwise_beta=pairwise_beta,
            forward_distance=forward_distance,
            turn_angle=turn_angle,
        )
        print("Initialization Complete")

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            top_p=1.0,
            use_cache=True,
            repetition_penalty = 1.05,
            num_return_sequences = self.num_generations
        )
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.reset_episode_state()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def build_prompt_inputs(self):
        texts = [self.processor.apply_chat_template( self.conversations, tokenize=False, add_generation_prompt=True)]

        image_inputs = []
        imgs, vids = process_vision_info(self.conversations)
        image_inputs.append(imgs)

        prompt_inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )

        prompt_inputs.to(self.device)
        return prompt_inputs

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

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
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")
        
    def reset_episode_state(self):
        self.topdown_map_list = []
        self.pending_action_list = []
        self.rgb_list = []
        self.step_logs = []
        self.decision_step = 0
        if self.stop_trace_enabled:
            self.trace_metadata_queue = []
            self.last_action_trace_metadata = None
        self.conversations = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }]

    def reset_episode(self, episode_id):
        """Restore base LoRA/optimizer and clear all episode-local state."""
        self.episode_id = str(episode_id)
        self.reset_episode_state()
        self.updater.reset_episode(self.episode_id)

    def finish_episode(self):
        if self.require_map and self.topdown_map_list:
            output_video_path = os.path.join(self.result_path, "video", f"{self.episode_id}.gif")
            imageio.mimsave(output_video_path, self.topdown_map_list)
        step_log_path = os.path.join(self.result_path, "log", f"steps_{self.episode_id}.json")
        with open(step_log_path, "w") as f:
            json.dump(self.step_logs, f, indent=2)

    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308,252))

        self.rgb_list.append(rgb_)
        # do not cut down rgb list while using uniform sampling
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            pending_before = len(self.pending_action_list)
            temp_action = self.pending_action_list.pop(0)
            if self.stop_trace_enabled:
                metadata = self.trace_metadata_queue.pop(0)
                metadata = {**metadata, "agent_action_source": "pending"}
                self.last_action_trace_metadata = {
                    "agent_action": temp_action,
                    "action_source": "pending",
                    "pending_before": pending_before,
                    "pending_after": len(self.pending_action_list),
                    "metadata": metadata,
                }
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}
        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url":f"data:image/jpeg;base64,{encode_image_base64(item)}"} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        update_result = self.updater.adapt_step(self.build_prompt_inputs())
        selected_actions = update_result["selected_actions"]
        decision_id = self.decision_step
        navigation = "TTRL medoid: " + ", ".join(
            self.action_id_to_str(action) for action in selected_actions
        )
        self.step_logs.append({
            "episode_id": self.episode_id,
            "decision_step": self.decision_step,
            "loss": update_result["loss"],
            "policy_loss": update_result["policy_loss"],
            "policy_loss_type": update_result["policy_loss_type"],
            "anchor_loss": update_result["anchor_loss"],
            "rewards": update_result["rewards"],
            "reward_std": update_result["reward_std"],
            "reward_margin": update_result["reward_margin"],
            "valid_count": update_result["valid_count"],
            "adapter_norm_before": update_result["adapter_norm_before"],
            "adapter_norm_after": update_result["adapter_norm_after"],
            "adapter_checksum_before": update_result["adapter_checksum_before"],
            "adapter_checksum_after": update_result["adapter_checksum_after"],
            "parameters_unchanged": update_result["parameters_unchanged"],
            "update_enabled": update_result["update_enabled"],
            "update_eligible": update_result["update_eligible"],
            "update_gate_reason": update_result["update_gate_reason"],
            "optimizer_step_executed": update_result["optimizer_step_executed"],
            "effective_parameter_change": update_result["effective_parameter_change"],
            "update_rolled_back": update_result["update_rolled_back"],
            "rollback_reason": update_result["rollback_reason"],
            "episode_update_opportunities": update_result["episode_update_opportunities"],
            "episode_optimizer_steps": update_result["episode_optimizer_steps"],
            "adapter_delta_norm": update_result["adapter_delta_norm"],
            "action_span_only": update_result["action_span_only"],
            "action_spans_available": update_result["action_spans_available"],
            "timing": update_result["timing"],
        })
        self.decision_step += 1
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        
        self.pending_action_list.extend(selected_actions)
        pending_before = len(self.pending_action_list)
        if self.stop_trace_enabled:
            self.trace_metadata_queue.extend(chunk_metadata(
                episode_id=str(self.episode_id),
                decision_id=decision_id,
                selected_chunk=selected_actions,
                raw_model_outputs=update_result["completions"],
                parsed_candidate_chunks=update_result["candidate_actions"],
                candidate_rewards=update_result["rewards"],
                selected_index=update_result["selected_candidate_index"],
            ))
        action = self.pending_action_list.pop(0)
        if self.stop_trace_enabled:
            metadata = self.trace_metadata_queue.pop(0)
            metadata = {**metadata, "agent_action_source": "generated"}
            self.last_action_trace_metadata = {
                "agent_action": action,
                "action_source": "generated",
                "pending_before": pending_before,
                "pending_after": len(self.pending_action_list),
                "metadata": metadata,
            }
        return {"action": action}

    def consume_action_trace_metadata(self):
        metadata = self.last_action_trace_metadata
        self.last_action_trace_metadata = None
        return metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,default=1,help="number of deterministic episode shards")
    parser.add_argument("--split-id",type=int,default=0,help="zero-based deterministic episode shard ID")
    parser.add_argument("--model-path",type=str,required=True,help="location of model weights")
    parser.add_argument("--lora-path",type=str,help="location of lora weights", default=None)
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--num-generations",type=int,help="number of sampled action chunks",default=8)
    parser.add_argument("--learning-rate",type=float,default=2e-6)
    parser.add_argument("--weight-decay",type=float,default=0.0)
    parser.add_argument("--max-grad-norm",type=float,default=1.0)
    parser.add_argument("--temperature",type=float,default=0.3)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--max-episodes",type=int,default=None,
                        help="limit this shard to a deterministic prefix for smoke tests")
    parser.add_argument("--episode-ids-file",type=str,default=None,
                        help="JSON episode ID list to select, in the file's order")
    parser.add_argument("--early-stop-rotation",type=int,default=25)
    parser.add_argument("--early-stop-steps",type=int,default=400,
                        help="zero permits one adaptation step before forced stop")
    parser.add_argument("--disable-update",action="store_true",
                        help="run full adaptation compute but skip optimizer.step")
    parser.add_argument("--stop-trace",action="store_true",
                        help="write default-off atomic action provenance JSONL")
    parser.add_argument("--official-navida-protocol",action="store_true",
                        help="match official NaVIDA prompt inputs; requires history 200")
    parser.add_argument("--update-policy",choices=("legacy", "gated"),default="legacy",
                        help="legacy always steps; gated requires reliable consensus")
    parser.add_argument("--min-reward-std",type=float,default=0.2)
    parser.add_argument("--min-reward-margin",type=float,default=0.25)
    parser.add_argument("--action-span-only",action="store_true",
                        help="apply the policy loss only to parsed answer/action tokens")
    parser.add_argument("--max-updates-per-episode",type=int,default=0,
                        help="zero disables the gated per-episode update limit")
    parser.add_argument("--anchor-l2-weight",type=float,default=0.0)
    parser.add_argument("--max-adapter-delta-norm",type=float,default=0.0,
                        help="zero disables rollback on excessive episode adapter drift")
    parser.add_argument("--pairwise-beta",type=float,default=1.0,
                        help="positive scale for gated best-vs-worst action-token loss")
    args = parser.parse_args()

    if args.split_num < 1 or not 0 <= args.split_id < args.split_num:
        raise ValueError("split-id must be in [0, split-num)")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("eval_ttt.py is single-process; launch ordinary Python processes, not torchrun")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("eval_ttt.py must not run with an initialized torch.distributed process group")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max-episodes must be positive when provided")
    if args.episode_ids_file is not None and args.max_episodes is not None:
        raise ValueError("episode-ids-file and max-episodes cannot be used together")
    if args.early_stop_rotation < 0 or args.early_stop_steps < 0:
        raise ValueError("early-stop limits must be non-negative")
    if args.min_reward_std < 0 or args.min_reward_margin < 0:
        raise ValueError("reward gate thresholds must be non-negative")
    if args.max_updates_per_episode < 0:
        raise ValueError("max-updates-per-episode must be non-negative")
    if args.anchor_l2_weight < 0 or args.max_adapter_delta_norm < 0:
        raise ValueError("adapter drift controls must be non-negative")
    if args.pairwise_beta <= 0:
        raise ValueError("pairwise-beta must be positive")
    if args.official_navida_protocol and args.max_action_history != 200:
        raise ValueError("official-navida-protocol requires --max-action-history 200")
    seed_all(args.seed)

    config = get_config(args.exp_config)
    with habitat.config.read_write(config):
        # self.config.habitat.task.measurements.success.success_distance=3.0
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
            
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    normalize_episode_ids(dataset)
    
    # Fixed modulo sharding is disjoint and independent of process timing.
    all_episodes = list(dataset.episodes)
    episode_ids_file_sha256 = None
    selection_seed = None
    if args.episode_ids_file is not None:
        requested_ids, selection_metadata, episode_ids_file_sha256 = load_episode_ids_file(
            args.episode_ids_file
        )
        all_episodes = select_episodes_in_file_order(all_episodes, requested_ids)
        if args.split_num > len(all_episodes):
            raise ValueError("episode selection would create one or more empty shards")
        expected_episode_ids = requested_ids
        selection_seed = selection_metadata.get("seed")
        dataset.episodes = shard_episodes(all_episodes, args.split_num, args.split_id)
        if not dataset.episodes:
            raise ValueError(f"selected episode shard {args.split_id} is unexpectedly empty")
    else:
        expected_episode_ids = [
            str(episode.episode_id)
            for shard_id in range(args.split_num)
            for episode in all_episodes[shard_id::args.split_num][:args.max_episodes]
        ]
        dataset.episodes = all_episodes[args.split_id::args.split_num][:args.max_episodes]
    write_episode_manifest(
        args.result_path, args.split_num, args.split_id, dataset.episodes, expected_episode_ids,
        episode_ids_file_sha256, selection_seed, config.habitat.dataset.split,
        {
            "official_navida_protocol": args.official_navida_protocol,
            "sequential_candidate_generation": args.official_navida_protocol,
            "lora_injection": (
                "native_transformers_adapter"
                if args.official_navida_protocol else "peft_model_wrapper"
            ),
            "max_action_history": args.max_action_history,
            "max_executed_chunks": 2,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "disable_update": args.disable_update,
            "stop_trace": args.stop_trace,
            "update_policy": args.update_policy,
            "min_reward_std": args.min_reward_std,
            "min_reward_margin": args.min_reward_margin,
            "action_span_only": args.action_span_only,
            "max_updates_per_episode": args.max_updates_per_episode,
            "anchor_l2_weight": args.anchor_l2_weight,
            "max_adapter_delta_norm": args.max_adapter_delta_norm,
            "pairwise_beta": args.pairwise_beta,
        },
    )

    evaluate_agent(config, args.split_id, dataset, args.model_path, args.lora_path, args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle, args.max_action_history,
                args.resolution_ratio, args.learning_rate, args.weight_decay, args.max_grad_norm,
                args.temperature, args.seed, args.max_episodes, args.early_stop_rotation,
                args.early_stop_steps, args.disable_update, args.stop_trace,
                args.official_navida_protocol, args.update_policy, args.min_reward_std,
                args.min_reward_margin, args.action_span_only, args.max_updates_per_episode,
                args.anchor_l2_weight, args.max_adapter_delta_norm, args.pairwise_beta)

if __name__ == "__main__":
    main()
