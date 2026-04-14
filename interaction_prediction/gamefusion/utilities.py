import glob
import numpy as np
from torch.utils.data import Dataset



class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vector_path = self.data_list[idx]
        data = np.load(vector_path)
        ego = data['ego'][0]
        neighbor = np.concatenate([data['ego'][1][np.newaxis, ...], data['neighbors']], axis=0)

        map_lanes = data['map_lanes'][:, :, :200:2]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]
        ego_future_states = data['gt_future_states'][0]
        neighbor_future_states = data['gt_future_states'][1]
        object_type = data['object_type']
        lidar_bev = data['lidar_bev'].astype(np.float32)
        lidar_bev = np.transpose(lidar_bev, (1, 0, 2, 3))
        camera_tokens = data['camera_tokens'].astype(np.float32)

        return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type, lidar_bev, camera_tokens