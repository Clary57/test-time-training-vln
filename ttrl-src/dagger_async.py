import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ['GLOG_minloglevel'] = '2'
import re
from tqdm import tqdm
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
from vllm import SamplingParams
import argparse, habitat
from habitat_baselines.config.default import get_config
from PIL import Image, ImageFont, ImageDraw
from qwen_vl_utils import process_vision_info
import multiprocessing as mp
import time
from openai import OpenAI
from vllm.multimodal.utils import encode_image_base64
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


SYSTEM_PROMPT = "You are a helpful assistant."

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

def setup_follower(sim):
    follower = ShortestPathFollower(
        sim,
        goal_radius=1.8,
        return_one_hot=False,
    )
    return follower

def evaluate_agent(result_queue,split_id, api_key, base_url, config, dataset, image_path, result_path, use_video, num_generations,
                    forward_distance, turn_angle, max_action_history, resolution_ratio) -> None:
 
    env = Env(config.TASK_CONFIG, dataset)
    os.makedirs(image_path, exist_ok=True)
    agent = NaVIDA_Agent(api_key, base_url, result_path, forward_distance, turn_angle, max_action_history, resolution_ratio, use_video, num_generations)

    num_episodes = len(env.episodes)

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0
    
    for _ in range(num_episodes):
        episode_start_time = time.time()
        obs = env.reset()
        follower = setup_follower(env.sim)

        reference_path = env.current_episode.reference_path
        next_waypoint_id = 1
        from_expert = False
        force_episode_end = False
        accumulated_error = 0
        action_seq = []
        actions = []

        left_expert_actions_num = 0
        episode_image_path = os.path.join(image_path,env.current_episode.episode_id)
        os.makedirs(episode_image_path, exist_ok=True)
        iter_step = 0

        agent.reset()
        rgb = obs["rgb"]
        if agent.resolution_ratio < 1:
            rgb = cv2.resize(rgb,(0,0),fx=agent.resolution_ratio,fy=agent.resolution_ratio)
        # rgb_ = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype('uint8')).convert('RGB')
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308,252))
        agent.rgb_list.append(rgb_)

        t_dict = {
            "t_episode": 0,
        }

        continuse_rotation_count = 0
        last_dtg = 999
        if os.path.exists(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id))):
            continue
        while not env.episode_over:
            
            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1

            if len(action_seq) == 0:
                if left_expert_actions_num > 0:
                    action = follower.get_next_action(reference_path[next_waypoint_id])
                    action_seq = [action]
                    left_expert_actions_num -= 1
                else:
                    if from_expert:
                        action = follower.get_next_action(reference_path[next_waypoint_id])
                        action_seq = [action]
                        # left_expert_actions_num  = random.randint(3,9) - 1
                        left_expert_actions_num  = 3 - 1
                    else:
                        action_seq = agent.act(obs, info, env.current_episode.episode_id)

            # if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
            #     action = {"action": 0}
            action = action_seq.pop(0)
            if action != follower.get_next_action(reference_path[next_waypoint_id]):
                accumulated_error += 1
            while follower.get_next_action(reference_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                from_expert = False
                left_expert_actions_num = 0
                if next_waypoint_id == len(reference_path) - 1:
                    follower = ShortestPathFollower(sim=env.sim, goal_radius=0.25, return_one_hot=False)
                if next_waypoint_id >= len(reference_path):
                    force_episode_end = True
                    action = 0
                    break
            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(reference_path)

            error_not_toleranted = ((from_expert == False and action == 0 and metrics["distance_to_goal"] >= 3.0) or (accumulated_error/max(1,int(60/(len(reference_path)-1))) > 0.8) or accumulated_error > 12)
            if wp_id_available and error_not_toleranted:
                from_expert = True
                accumulated_error = 0
                action = follower.get_next_action(reference_path[next_waypoint_id])
            
            # action check
            if action == 0 and not force_episode_end:
                action = follower.get_next_action(reference_path[next_waypoint_id])
            
            iter_step+=1
            obs = env.step(action)
            rgb = obs["rgb"]
            if agent.resolution_ratio < 1:
                rgb = cv2.resize(rgb,(0,0),fx=agent.resolution_ratio,fy=agent.resolution_ratio)
            # rgb_ = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype('uint8')).convert('RGB')
            rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
            rgb_ = rgb_.resize((308,252))
            agent.rgb_list.append(rgb_)
            # do not cut down rgb list while using uniform sampling
            if len(agent.rgb_list) > agent.max_action_history:
                agent.rgb_list = agent.rgb_list[1:]
            actions.append(action)
            if env.episode_over or force_episode_end:            
                break
            
        episode_save = metrics["distance_to_goal"] <= 3.0 and metrics["path_length"] <= 20 and env.episode_over
        if episode_save:
            assert len(actions) == len(agent.rgb_list) - 1
            assert actions[-1] == 0
            with open(os.path.join(result_path, f"dagger_{split_id}.jsonl"), 'a', encoding='utf-8') as f:
                result = {
                    'actions':actions,
                    'episode_id':env.current_episode.episode_id
                }
                f.write(json.dumps(result) + "\n")
            os.makedirs(episode_image_path, exist_ok=True)
            for i ,item in enumerate(agent.rgb_list):
                current_frame_name = os.path.join(episode_image_path,'frame_'+str(i)+'.jpg')
                item.save(current_frame_name)

            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1

        # with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
        #     json.dump(result_dict, f, indent=4)
        
        t_dict["t_episode"] = time.time() - episode_start_time
        result_queue.put(t_dict)

class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance, 
                    turn_angle, max_action_history, resolution_ratio, use_video, num_generations = 1, require_map=True):
        
        print("Initialize NaVid")
        
        self.result_path = result_path
        self.require_map = require_map
        self.use_video = use_video
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        os.makedirs(self.result_path, exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id
        
        self.sampling_params = SamplingParams(
            n = 1,
            temperature=0.2,
            max_tokens=512, # i.e. max_completion_tokens
        )

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})

        self.count_id = 0
        self.reset()

    def uniform_sample_with_ends(self, data, n):
        # n > 2
        if len(data) <= n:
            return data

        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]

    def predict_inference(self):

        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        
        return output_text

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
        self.topdown_map_list = []
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.rgb_list = []

        self.first_forward = False
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        # rgb = observations["rgb"]
        # if self.resolution_ratio < 1:
        #     rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        # # rgb_ = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype('uint8')).convert('RGB')
        # rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        # rgb_ = rgb_.resize((308,252))
        # self.rgb_list.append(rgb_)
        # # do not cut down rgb list while using uniform sampling
        # if len(self.rgb_list) > self.max_action_history:
        #     self.rgb_list = self.rgb_list[1:]

        # if self.require_map:
        #     top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
        #     output_im = np.concatenate((rgb, top_down_map), axis=1)

        # if len(self.pending_action_list) != 0 :
        #     temp_action = self.pending_action_list.pop(0)
            
        #     if self.require_map:
        #         img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
        #         self.topdown_map_list.append(img)
        #     return {"action": temp_action}

        # for observation1+observation2 action style
        self.conversations = self.conversations[:1]
        content = []
        # if self.use_video:
        #     content.append({"type": "video", "video": self.rgb_list})
        # else:
        #     content.extend([{"type": "image", "image": item} for item in self.rgb_list])
        # content.append({"type": "text", "text": self.promt_template.format(observations["instruction"]["text"])+answer_template})

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            if self.use_video:
                content.append({"type": "video", "video": self.uniform_sample_with_ends(self.rgb_list[:-1],8)})
            else:
                content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in self.uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
            # content.extend([{"type": "image", "image": item} for item in self.uniform_sample_with_ends(self.rgb_list,8)])
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})


        self.conversations.append({
                "role": "user",
                "content": content
            })

        navigation = self.predict_inference()
        
        # if self.require_map:
        #     img = self.addtext(output_im, observations["instruction"]["text"], navigation)
        #     self.topdown_map_list.append(img)

        # action_index, numeric = self.extract_result(navigation)
        
        result = self.extract_multi_result(navigation)
        # result = result[:3]
        self.pending_action_list = []
        for action_index,numeric in result:

            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric/self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,round(numeric/self.turn_angle))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list)==0:
                print('random select an action')
                action_index = random.randint(1, 3)
                navigation = self.action_id_to_str(action_index)
                self.pending_action_list.append(action_index)


        # return {"action": self.pending_action_list.pop(0)}
        return self.pending_action_list


def main():
    seed_all()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str,required=True,help="path to config yaml containing info about experiment")
    parser.add_argument("--split-num",type=int,required=True,help="chunks of evluation")
    parser.add_argument("--resolution-ratio",type=float,help="location of model weights",default=0.5)
    parser.add_argument("--image-path",type=str,required=True,help="location to save images")
    parser.add_argument("--result-path",type=str,required=True,help="location to save results")
    parser.add_argument("--forward-distance",type=int,help="distance that one forward action takes",default=25)
    parser.add_argument("--turn-angle",type=int,help="angle that one turn action takes",default=15)
    parser.add_argument("--max-action-history",type=int,help="the maximum num of action history",default=10)
    parser.add_argument("--use-video",type=str2bool,help="whether use video or multi image",default=False)
    parser.add_argument("--num-generations",type=int,help="whether use video or multi image",default=1)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
            
    dataset = habitat.datasets.make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    
    dataset_splits = dataset.get_splits(args.split_num,allow_uneven_splits=True)

    # evaluate_agent(config, args.split_id, dataset_split, args.model_path, args.lora_path, args.result_path,
    #             args.use_video, args.num_generations, args.forward_distance, args.turn_angle, args.max_action_history,
    #             args.resolution_ratio)

    num_episodes = len(dataset.episodes) 

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (result_queue, i, api_key, base_url, config, dataset_splits[i], args.image_path, args.result_path,
                args.use_video, args.num_generations, args.forward_distance, args.turn_angle, 
                args.max_action_history, args.resolution_ratio)
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()