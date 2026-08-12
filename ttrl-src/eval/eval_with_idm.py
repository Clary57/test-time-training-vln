import json
from matplotlib.pylab import True_
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os, io
import re
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
from peft import PeftModel
from vllm.multimodal.utils import encode_image_base64
import habitat_sim
from scipy.spatial.transform import Rotation as R

SYSTEM_PROMPT = "You are a helpful assistant."


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


def seed_all():
    np.random.seed(41)
    random.seed(41)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def jpeg_simulate(image: Image.Image, quality: int = 75) -> Image.Image:

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def evaluate_agent(config, split_id, dataset, model_path, lora_path, result_path, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio) -> None:
 
    env = Env(config.habitat, dataset)

    agent = NaVid_Agent(model_path, lora_path, result_path, forward_distance, turn_angle, max_action_history, resolution_ratio, num_generations)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 400

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success", "ndtw"}

    count = 0
    
    for _ in trange(num_episodes):
        obs = env.reset()
        iter_step = 0
        agent.reset()

        continuse_rotation_count = 0
        last_dtg = 999
        # if os.path.exists(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id))):
        #     continue
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            action = agent.act(obs, info, env.current_episode.episode_id)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        result_dict['idm_score'] = agent.idm_score
        count+=1

        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)

class NaVid_Agent(Agent):
    def __init__(self, model_path, lora_path, result_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, num_generations = 1, require_map=True):
        
        print("Initialize NaVIDA")
        
        self.result_path = result_path
        self.require_map = require_map
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["use_cache"] = True
        model_init_kwargs['torch_dtype'] = torch.bfloat16
        if 'navida' in model_path.lower():
            from src.model.navida_with_adapter import NaVIDAWithAdapterQwen2_5_VLForConditionalGeneration
            from src.model.navida_pd import NaVIDAPDQwen2_5_VLForConditionalGeneration
            from src.model.navida import NaVIDAQwen2_5_VLForConditionalGeneration
            self.model = NaVIDAQwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)
        elif "qwen2_vl" in model_path.lower() or "qwen2-vl" in model_path.lower():
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)
        elif "qwen2.5_vl" in model_path.lower() or "qwen2.5-vl" in model_path.lower():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_init_kwargs)
        else:
            raise ValueError(
                    "No model init!"
                )
        if lora_path is not None and lora_path!= '':
            print('Loading LoRA weights...')
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print('Merging LoRA weights...')
            self.model = self.model.merge_and_unload()
            print('Model is loaded...')

        self.device = 'cuda'
        self.model.to(self.device)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_pixels = 501760
        print("Initialization Complete")

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."

        self.idm_prompt_template = "Imagine you are a robot programmed for navigation tasks. "\
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
            num_return_sequences = self.num_generations
        )
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        
        self.last_idm_rgb = None
        self.last_gt_action = None
        self.idm_score = []
        self.sim = make_sim()
        self.agent = self.sim.initialize_agent(0)
        self.initial_state = self.agent.get_state()

        self.count_id = 0
        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]


    def predict_inference(self):
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
        with torch.inference_mode():
            outputs = self.model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=True,
                # output_scores=True,
                # return_dict_in_generate=True,
                )
        # output_ids = outputs.sequences
        output_ids = outputs
        input_token_len = prompt_inputs["input_ids"].shape[1]
        n_diff_input_output = (prompt_inputs["input_ids"] != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs_text = self.processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # outputs_text = self.processor.batch_decode(output_ids[:, :], skip_special_tokens=False)
        outputs_text = outputs_text[0]
        outputs_text = outputs_text.strip()
        return outputs_text
    

    def extract_multi_result_action(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            if action_index == 1:
                num = self.forward_distance
            elif action_index == 0:
                num = None
            else:
                num = self.turn_angle
            if num is not None:
                result.extend([action_index]*int(numeric/num))
            else:
                result.append(0)
        return result

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
        
    def reset(self):       
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_tensor = None
        self.entropy_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id = 0
        self.count_stop = 0
        self.pending_action_list = []
        self.rgb_list = []
        self.chunk_rgb_list = []

        self.first_forward = False
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        
        self.last_idm_rgb = None
        self.last_gt_action = None
        self.idm_score = []

        
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
        
        self.chunk_rgb_list.append(len(self.rgb_list)-1)
        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            # content.extend([{"type": "image_url", "image_url":f"data:image/jpeg;base64,{encode_image_base64(item)}"} for item in self.sample_with_entropy(self.rgb_list[:-1],8)])
            content.extend([{"type": "image_url", "image_url":f"data:image/jpeg;base64,{encode_image_base64(item)}"} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
            # content.extend([{"type": "image", "image": item} for item in self.uniform_sample_with_ends(self.rgb_list,8)])
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})

        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        self.count_id += 1
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)
        
        result = self.extract_multi_result(navigation)

        # select_action_idx = 2

        # result = result[:select_action_idx]
        if self.last_gt_action is not None:
            self.idm_score.append(self.act_idm(self.last_idm_rgb, rgb_))

        self.last_gt_action = navigation
        self.last_idm_rgb = rgb_
        
        for idx, (action_index,numeric) in enumerate(result):
            pending_action_list = []
            if action_index == 0:
                pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric/self.forward_distance))):
                    pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    pending_action_list.append(3)
            
            if action_index is None or len(pending_action_list)==0:
                print('random select an action')
                action_index = random.randint(1, 3)
                navigation = self.action_id_to_str(action_index)
                pending_action_list.append(action_index)
            
            self.pending_action_list.extend(pending_action_list)
        self.last_idm_rgb = self.rgb_list[-1]
        return {"action": self.pending_action_list.pop(0)}
    
    def calculate_acc(self, prediction, gt_answer):
        # return 1 if prediction == gt_answer else 0
        prediction_action = self.extract_multi_result_action(prediction)
        gt_answer_action = self.extract_multi_result_action(gt_answer)
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
    
    def act_idm(self, img0, img1):
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given an image of current view'})
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(img0)}"})
        content.append({"type": "text", "text": 'and an image of the goal view'})
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(img1)}"})
        item = "Analyze the two images to predict the navigation action that would move the robot from the current viewpoint to the goal view, "\
                "which could involve turning left "\
                "or right by a specific degree or moving forward a certain distance."
        content.append({"type": "text", "text": item})

        self.conversations.append({
                "role": "user",
                "content": content
            })

        prediction = self.predict_inference()
        gt_answer = self.last_gt_action
        # print(f"prediction = {prediction}, gt = {gt_answer}")
        return self.calculate_acc(prediction, gt_answer)


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--split-id",type=int,required=True,help="chunks ID of evluation")
    parser.add_argument("--model-path",type=str,required=True,help="location of model weights")
    parser.add_argument("--lora-path",type=str,help="location of lora weights", default=None)
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--num-generations",type=int,help="whether use video or multi image",default=1)
    args = parser.parse_args()

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
    
    dataset_split = dataset.get_splits(args.split_num,allow_uneven_splits=True)[args.split_id]

    evaluate_agent(config, args.split_id, dataset_split, args.model_path, args.lora_path, args.result_path,
                args.num_generations, args.forward_distance, args.turn_angle, args.max_action_history,
                args.resolution_ratio)

if __name__ == "__main__":
    main()