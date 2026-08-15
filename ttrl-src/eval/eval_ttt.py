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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainer.trajectory_consensus_updater import TrajectoryConsensusUpdater

SYSTEM_PROMPT = "You are a helpful assistant."


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


def write_episode_manifest(result_path, split_num, split_id, episodes, expected_episode_ids):
    """Persist this shard and the common expected ID set before resume filtering."""
    os.makedirs(result_path, exist_ok=True)
    manifest = {
        "protocol": "episode-local-step-wise-ttrl",
        "split_num": split_num,
        "split_id": split_id,
        "episode_ids": [str(episode.episode_id) for episode in episodes],
        "expected_episode_ids": expected_episode_ids,
    }
    with open(os.path.join(result_path, "episode_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def evaluate_agent(config, split_id, dataset, model_path, lora_path, result_path, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio,
                    learning_rate, weight_decay, max_grad_norm, temperature, seed,
                    max_episodes=None, early_stop_rotation=25, early_stop_steps=400) -> None:
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
                        seed)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = early_stop_rotation
    EARLY_STOP_STEPS = early_stop_steps

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success", "ndtw"}

    count = 0
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    for _ in trange(num_episodes):
        obs = env.reset()
        iter_step = 0
        agent.reset_episode(env.current_episode.episode_id)

        continuse_rotation_count = 0
        last_dtg = 999
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": 0}
            else:
                action = agent.act(obs, info, env.current_episode.episode_id)

            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1

        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)
        agent.finish_episode()

class NaVIDA_Agent(Agent):
    def __init__(self, model_path, lora_path, result_path, forward_distance,
                    turn_angle, max_action_history, resolution_ratio, num_generations,
                    learning_rate, weight_decay, max_grad_norm, temperature, seed,
                    require_map=True):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.require_map = require_map
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        self.seed = seed
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["use_cache"] = True
        model_init_kwargs['torch_dtype'] = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)

        if lora_path is not None and lora_path != '':
            print('Loading trainable LoRA weights...')
            self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=True)
        else:
            print('Creating a new trainable LoRA adapter...')
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=8,
                    lora_alpha=8,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    exclude_modules=".*visual.*",
                ),
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
            temp_action = self.pending_action_list.pop(0)
            
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
        navigation = "TTRL medoid: " + ", ".join(
            self.action_id_to_str(action) for action in selected_actions
        )
        self.step_logs.append({
            "episode_id": self.episode_id,
            "decision_step": self.decision_step,
            "loss": update_result["loss"],
            "rewards": update_result["rewards"],
            "reward_std": update_result["reward_std"],
            "valid_count": update_result["valid_count"],
            "adapter_norm_before": update_result["adapter_norm_before"],
            "adapter_norm_after": update_result["adapter_norm_after"],
            "timing": update_result["timing"],
        })
        self.decision_step += 1
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        
        self.pending_action_list.extend(selected_actions)

        return {"action": self.pending_action_list.pop(0)}


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
    parser.add_argument("--early-stop-rotation",type=int,default=25)
    parser.add_argument("--early-stop-steps",type=int,default=400,
                        help="zero permits one adaptation step before forced stop")
    args = parser.parse_args()

    if args.split_num < 1 or not 0 <= args.split_id < args.split_num:
        raise ValueError("split-id must be in [0, split-num)")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("eval_ttt.py is single-process; launch ordinary Python processes, not torchrun")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("eval_ttt.py must not run with an initialized torch.distributed process group")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max-episodes must be positive when provided")
    if args.early_stop_rotation < 0 or args.early_stop_steps < 0:
        raise ValueError("early-stop limits must be non-negative")
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
    
    # Fixed modulo sharding is disjoint and independent of process timing.
    all_episodes = list(dataset.episodes)
    expected_episode_ids = [
        str(episode.episode_id)
        for shard_id in range(args.split_num)
        for episode in all_episodes[shard_id::args.split_num][:args.max_episodes]
    ]
    dataset.episodes = all_episodes[args.split_id::args.split_num][:args.max_episodes]
    write_episode_manifest(
        args.result_path, args.split_num, args.split_id, dataset.episodes, expected_episode_ids
    )

    evaluate_agent(config, args.split_id, dataset, args.model_path, args.lora_path, args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle, args.max_action_history,
                args.resolution_ratio, args.learning_rate, args.weight_decay, args.max_grad_norm,
                args.temperature, args.seed, args.max_episodes, args.early_stop_rotation,
                args.early_stop_steps)

if __name__ == "__main__":
    main()
