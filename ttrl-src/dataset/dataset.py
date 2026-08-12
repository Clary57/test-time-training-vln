from torch.utils.data.dataset import Dataset
import gzip
import json
import random


class TTTRLDataset(Dataset):
    def __init__(
        self, 
        config_path: str = './R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz'
    ):
        super().__init__()
        self.config_path = config_path
        with gzip.open(config_path) as f:
            data = f.read()
        data = json.loads(data)

        self.episodes_list = []

        for item in data['episodes']:
            episode_id = int(item['episode_id'])
            self.episodes_list.append(episode_id)
            
        random.shuffle(self.episodes_list)

    def __getitem__(self, index):
        # index = 891
        item = {
            'episode_id': self.episodes_list[index]
        }
        return item

    def __len__(self):
        return len(self.episodes_list)