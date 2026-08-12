import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange, tqdm
import os, io
import re
import torch
import cv2
import imageio
import random
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
import argparse
from PIL import Image
from qwen_vl_utils import process_vision_info
from collections import Counter
from peft import PeftModel
from vllm.multimodal.utils import encode_image_base64
from src.model.navida import NaVIDAQwen2_5_VLForConditionalGeneration
import habitat_sim
from scipy.spatial.transform import Rotation as R


SYSTEM_PROMPT = "You are a helpful assistant."

def seed_all():
    np.random.seed(41)
    random.seed(41)

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [
        lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(n)
    ]

def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    # sim_cfg.scene_id = None

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # 定义离散动作
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }

    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    )

class NaVIDA_Agent():
    def __init__(self, model_path):
        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["use_cache"] = True
        model_init_kwargs['torch_dtype'] = torch.bfloat16
        self.model = NaVIDAQwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)
        self.device = 'cuda'
        self.model.to(self.device)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_pixels = 501760
        print("Initialization Complete")

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
    "You have been given an image of current view and an image of the goal view. "\
    "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "\
    "which could involve turning left "\
    "or right by a specific degree or moving forward a certain distance."
        
        self.generation_config = GenerationConfig(
            do_sample=True,  
            temperature=0.2,
            max_new_tokens=512,
            top_p=1.0,
            use_cache=True,
            num_return_sequences = 1
        )
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)
        self.initial_state = self.agent.get_state()
    
    def predict_inference(self):
        texts = [self.processor.apply_chat_template( self.conversations, tokenize=False, add_generation_prompt=True)]

        image_inputs = []
        imgs, vids = process_vision_info(self.conversations)
        imgs = [item.resize((308,252)) for item in imgs]
        image_inputs.append(imgs)

        prompt_inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )

        prompt_inputs.to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
                )
        # output_ids = outputs.sequences
        output_ids = outputs
        input_token_len = prompt_inputs["input_ids"].shape[1]
        n_diff_input_output = (prompt_inputs["input_ids"] != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs_text = self.processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs_text = outputs_text[0]
        outputs_text = outputs_text.strip()
        return outputs_text

    def act(self, example):
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given an image of current view'})
        content.append({"type": "image", "image": example["conversations"][0]["image"][0]} )
        content.append({"type": "text", "text": 'and an image of the goal view'})
        content.append({"type": "image", "image": example["conversations"][0]["image"][1]} )
        item = example["conversations"][0]["value"].split('goal view. ')
        content.append({"type": "text", "text": item[1]})

        self.conversations.append({
                "role": "user",
                "content": content
            })

        prediction = self.predict_inference()
        gt_answer = example["conversations"][1]["value"]
        # print(f"prediction = {prediction}, gt = {gt_answer}")
        return self.calculate_acc(prediction, gt_answer)
    
    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            if action_index == 1:
                num = 25
            elif action_index == 0:
                num = None
            else:
                num = 15
            if num is not None:
                result.extend([action_index]*int(numeric/num))
            else:
                result.append(0)
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
    
    def calculate_acc(self, prediction, gt_answer):
        # return 1 if prediction == gt_answer else 0
        prediction_action = self.extract_multi_result(prediction)
        gt_answer_action = self.extract_multi_result(gt_answer)
        action_map = {
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
        }
        def quaternion_to_yaw(q):
            rotation = np.array([q.w, q.x, q.y, q.z])
            # camera2world
            rotation_matrix = R.from_quat(rotation[[1, 2, 3, 0]]).as_matrix()
            yaw = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
            return yaw

        def rollout(seq):
            self.agent.set_state(self.initial_state)
            for a in seq:
                if a != 0:
                    self.sim.step(action_map[a])
            position = self.agent.get_state().position.copy()
            yaw = quaternion_to_yaw(self.agent.get_state().rotation)
            return position, yaw

        def linear_score(d, d_max):
            return max(0.0, 1.0 - d / d_max)
        
        def exp_score(d, sigma):
            return np.exp(-d/sigma)
            # return np.exp(- (d**2) / (sigma**2))

        pos1, yaw1 = rollout(prediction_action)
        pos2, yaw2 = rollout(gt_answer_action)

        delta_pos = np.linalg.norm(pos1 - pos2)
        delta_yaw = abs(yaw1-yaw2)
        delta_yaw = min(delta_yaw, 2*np.pi - delta_yaw)
        assert delta_yaw >= 0

        score = exp_score(delta_pos, 0.1) * exp_score(delta_yaw, np.pi/12)
        # score = score / 2

        return score




def evaluate_idm(split_id, dataset, model_path, result_path):
    agent = NaVIDA_Agent(model_path)
    acc = 0
    total_count = 0
    pbar = tqdm(dataset)
    for example in pbar:
        acc += agent.act(example)
        total_count += 1
        pbar.set_description(f"{acc:.4f}/{total_count} = {acc/total_count:.4f}")
    acc_total = acc / len(dataset)
    print(f"total acc = {acc_total}")
    result_dict = {
        'split_id': split_id,
        'total_count':total_count,
        'acc':acc
    }
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, "log"), exist_ok=True)
    with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(split_id)), "w") as f:
        json.dump(result_dict, f, indent=4)

def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno-path",type=str,required=True,help="path to annotation")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,required=True,help="chunks ID of evluation")
    parser.add_argument("--model-path",type=str,required=True,help="location of model weights")
    parser.add_argument("--result-path",type=str,)
    args = parser.parse_args()

    annotations = []
    with open(args.anno_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            annotations.append(item)

    dataset_split = split_list(annotations, args.split_num)[args.split_id]

    evaluate_idm(args.split_id, dataset_split, args.model_path,args.result_path)

if __name__ == "__main__":
    main()