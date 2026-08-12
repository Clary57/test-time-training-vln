from habitat_baselines.config.default import get_config
import habitat
from vllm.multimodal.utils import encode_image_base64
import cv2
from PIL import Image

SYSTEM_PROMPT = "You are a helpful assistant."

def init_env(simulator_config_path, local_rank):
    config = get_config(simulator_config_path)

    with habitat.config.read_write(config):
        config.habitat.simulator.habitat_sim_v0.update(
            {
                "gpu_device_id": local_rank,
            }
        )
    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    env = habitat.Env(config.habitat, dataset)
    mapping = {int(env._dataset.episodes[i].episode_id):i for i in range(len(env._dataset.episodes))}
    return env, mapping

def uniform_sample_with_ends(data, n):
    # n > 2
    if len(data) <= n:
        return data

    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]

class Episode_State:
    def __init__(
        self, 
        simulator_config_path,
        local_rank,
    ):
        self.alive = True
        self.local_rank = local_rank

        self.rgb_list = []
        self.conversations = []
        self.pending_actions = []
        self.episode_id = None
        self.instruction = None

        self.step = 0

        self.env, self.mapping = init_env(simulator_config_path, local_rank)

        self.promt_template = "Imagine you are a robot programmed for navigation tasks. "\
            "You have been given a video of historical observations and an image of the current observation. "\
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "\
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        
        self.target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success", "ndtw"}

        self.resolution_ratio = 0.5
    
    def reset(self, episode_id: int):
        observations = self.reset_env_with_episode_id(self.mapping[int(episode_id)])
        self.alive = True
        self.step = 0

        rgb = observations["rgb"]
        rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        rgb_ = rgb_.resize((308,252))
        self.rgb_list = [rgb_]

        self.conversations = self.conversations[:1]
        self.episode_id = int(episode_id)
        self.instruction = observations["instruction"]["text"]
        # print(f"************** rank = {self.local_rank}  episode_id = {episode_id} *********************8")
        return observations

    def reset_env_with_episode_id(self, i):
        '''
        Reset the habitat envrioment with `episode_id = i`
        '''
        self.env._reset_stats()
        assert len(self.env.episodes) > 0, "Episodes list is empty"
        if self.env._current_episode is not None:
            self.env._current_episode._shortest_path_cache = None

        self.env._current_episode = self.env._dataset.episodes[i]
        self.env._episode_from_iter_on_reset = True
        self.env._episode_force_changed = False
        
        self.env.reconfigure(self.env._config)

        observations = self.env.task.reset(episode=self.env.current_episode)
        self.env._task.measurements.reset_measures(
            episode=self.env.current_episode, task=self.env.task
        )
        return observations
    
    def act(self, actions):
        '''
        Execute actions
        '''
        for action in actions:
            if not self.alive:
                break
            new_observation = self.env.step(action)

            rgb = new_observation["rgb"]
            rgb = cv2.resize(rgb,(0,0),fx=self.resolution_ratio,fy=self.resolution_ratio)
            rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
            rgb_ = rgb_.resize((308,252))
            self.rgb_list.append(rgb_)

            # self.alive = False if action == 0 else self.alive
            self.alive = False if self.env.episode_over else self.alive
        self.step += 1
    
    def build_prompt(self):
        if not self.alive:
            return self.conversations
        
        self.conversations = self.conversations[:1]

        content = []
        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url":f"data:image/jpeg;base64,{encode_image_base64(item)}"} for item in uniform_sample_with_ends(self.rgb_list[:-1],8)])
        else:
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"})
        item = self.promt_template.format(self.instruction).split('current observation')
        content.append({"type": "text", "text": item[1]})

        self.conversations.append({
                "role": "user",
                "content": content
            })
        
        return self.conversations
    
    def get_metrics(self):
        assert not self.alive, f"You can only access the metrics when the episode is over in current episode_id {self.episode_id}, with len(images) = {len(self.rgb_list)}, step = {self.step}"
        info = self.env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in self.target_key if k in info}
        result_dict["id"] = self.episode_id
        return result_dict
        
            
