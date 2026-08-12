import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ['GLOG_minloglevel'] = '2'
import json
import PIL.Image as Image
from tqdm import tqdm
from habitat_extensions import measures, task
import multiprocessing as mp
import time
import torch
import numpy as np
import argparse
from transformers.models.dinov2_with_registers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
from transformers import AutoProcessor


def spilt_method_1(x):
    return x.split("_")[1].split(".")[0]

def spilt_method_2(x):
    return x.split(".")[0]

CONFIG = {
    'r2r': {
        'image_path': '/nvme-ssd1/zwy/data/images_downsample/R2R_train',
        'annotation_path': '/nvme-ssd1/zwy/lam_vln/data/sub_dataset/r2r.jsonl',
        'split_method': spilt_method_1,
    },
    'rxr': {
        'image_path': '/nvme-ssd1/zwy/data/images_downsample/RxR_train',
        'annotation_path': '/nvme-ssd1/zwy/lam_vln/data/sub_dataset/rxr.jsonl',
        'split_method': spilt_method_1,
    },
    'envdrop': {
        'image_path': '/nvme-ssd1/zwy/data/images_downsample/R2R_envdrop',
        'annotation_path': '/nvme-ssd1/zwy/lam_vln/data/sub_dataset/envdrop.jsonl',
        'split_method': spilt_method_1,
    },
    'scalevln': {
        'image_path': '/nvme-ssd1/zwy/data/images_downsample/scale_vln',
        'annotation_path': '/nvme-ssd1/zwy/lam_vln/data/sub_dataset/scalevln.jsonl',
        'split_method': spilt_method_2,
    },
    'dagger_r2r': {
        'image_path': '/nvme-ssd1/zwy/data/images_downsample/dagger_r2r',
        'annotation_path': '/nvme-ssd1/zwy/lam_vln/data/sub_dataset/dagger_wo_rxr.jsonl',
        'split_method': spilt_method_1,
    }
}

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [
        lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(n)
    ]

def get_embedding(dino_model, x):
    with torch.no_grad():
        outputs = dino_model(x)
    sequence_output = outputs[0]
    cls_token = sequence_output[:, 0]
    return cls_token

def extract_data(
        result_queue, 
        annotations, 
        image_path = None,
        split_method = None,
        dino_name_or_path = None,
        device_id = None
    ):
    device = torch.device(f"cuda:{device_id}")
    dino_model = Dinov2WithRegistersModel.from_pretrained(dino_name_or_path).to(device)
    dino_model.eval()
    processor = AutoProcessor.from_pretrained(
            dino_name_or_path
        )
    for idx, annotation in enumerate(annotations):
        output_dict = {
            "time_per_episode": 0,
        }
        episode_start_time = time.time()
        episode_id = int(annotation['episode_id'])
        reference_actions = annotation["actions"]
        video_id = annotation["video_id"]

        episode_image_path = os.path.join(image_path, str(video_id))
        image_name_list = os.listdir(episode_image_path)
        image_name_list = sorted(image_name_list, key=lambda x: int(split_method(x)))
        image_name_list = [episode_image_path+'/'+image for image in image_name_list]

        cos_sim_list = []
        for start_id, start_image_name in enumerate(image_name_list[:-1]):
            current_cos_sim_list = []
            start_frame = processor([Image.open(start_image_name)], return_tensors="pt",)['pixel_values'].to(device)
            start_frame_embedding = get_embedding(dino_model, start_frame)
            for end_id, end_image_name in enumerate(image_name_list[start_id+1:]):
                end_frame = processor([Image.open(end_image_name)], return_tensors="pt",)['pixel_values'].to(device)
                end_frame_embedding = get_embedding(dino_model, end_frame)
                cos_similarity = torch.nn.functional.cosine_similarity(start_frame_embedding, end_frame_embedding, dim=-1, eps=1e-8)
                current_cos_sim_list.append(cos_similarity.item())
            cos_sim_list.append(current_cos_sim_list)
        output_dict["cos_sim_list"] = cos_sim_list
        output_dict["time_per_episode"] = time.time() - episode_start_time
        output_dict["episode_id"] = episode_id
        result_queue.put(output_dict)


         


    # env = habitat.Env(env_config.habitat, dataset)

    # assert len(annotations) == len(env.episodes)
    # print(f"total episode = {len(annotations)}")

    # for idx, episode in enumerate(tqdm(env.episodes)):
    #     output_dict = {
    #         "time_per_episode": 0,
    #     }
    #     episode_start_time = time.time()
    #     env.current_episode = episode
    #     observation = env.reset()
    #     annotation = annotations[idx]
    #     assert annotation['episode_id'] == int(episode.episode_id)
    #     reference_actions = annotation["actions"]
    #     step_id = 0 
    #     curren_image_path = os.path.join(image_path,str(episode.episode_id))
    #     os.makedirs(curren_image_path, exist_ok=True)
    #     pose_list = []
    #     while not env.episode_over:
    #         rgb = observation["rgb"]
    #         rgb_frame = Image.fromarray(rgb).convert("RGB")
    #         rgb_frame = rgb_frame.resize((320,240))
    #         rgb_frame.save(os.path.join(curren_image_path, f"{step_id}.jpg"))
    #         current_pose = pose_to_matrix(env.sim.get_agent_state()) # body2world
    #         pose_list.append(current_pose.tolist())
    #         action = reference_actions.pop(0) 
    #         observation = env.step(action)  
    #         step_id += 1
    #     output_dict["time_per_episode"] = time.time() - episode_start_time
    #     output_dict["episode_id"] = int(episode.episode_id)
    #     output_dict["poses"] = pose_list
    #     result_queue.put(output_dict)
    # env.close()

def process_single_dataset(dataset_name, dino_name_or_path, device_num, split_num):

    ANNOT_PATH = CONFIG[dataset_name]['annotation_path']
    IMAGE_PATH = CONFIG[dataset_name]['image_path']
    SPLIT_METHOD = CONFIG[dataset_name]['split_method']

    annotations = []
    with open(ANNOT_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            annotations.append(item)
    annotations = sorted(annotations, key=lambda x: x["episode_id"])
    falici = {annotations[i]['episode_id']:i for i in range(len(annotations))}
    
    num_episodes = len(annotations)
    sub_annotations = split_list(annotations, device_num*split_num)

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for device_id in range(device_num):
        for i in range(split_num):
            worker_id = device_id * split_num + i
            worker_args = (
                    result_queue,
                    sub_annotations[worker_id],
                    IMAGE_PATH,
                    SPLIT_METHOD,
                    dino_name_or_path,
                    device_id,
                )
            p = mp.Process(target=extract_data, args=worker_args, daemon=True)
            p.start()
            processes.append(p)

    with tqdm(total=num_episodes, desc="generating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            episode_id = result['episode_id']
            annotations[falici[int(episode_id)]]['cos_sim_list'] = result['cos_sim_list']
            pbar.update(1)
            pbar.set_postfix(time_per_episode = result['time_per_episode'])
    
    for p in processes:
        p.join()
    
    os.makedirs('./dino_sim', exist_ok=True)
    with open(os.path.join('./dino_sim',f"{dataset_name}.jsonl"), 'w', encoding='utf-8') as f:
        for item in annotations:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(len(annotations))
    

def main(dataset2process, dino_name_or_path, device_num, num_thread):
    for dataset_name in dataset2process:
        process_single_dataset(dataset_name, dino_name_or_path, device_num, num_thread)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r",],
    )
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--dino_name_or_path", type=str, default="")
    
    args = parser.parse_args()
    print(args.dataset_name)
    main(args.dataset_name, args.dino_name_or_path, args.device_num, args.num_thread)