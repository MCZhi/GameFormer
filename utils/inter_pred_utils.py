import os
import glob
import torch
import logging
import numpy as np
import tensorflow as tf

from torch.utils.data import Dataset
from torch.nn import functional as F
from google.protobuf import text_format

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego'][0]
        neighbor = np.concatenate([data['ego'][1][np.newaxis, ...], data['neighbors']], axis=0)[:21, :, :]

        map_lanes = data['map_lanes'][:, :, :200:2]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]
        ego_future_states = data['gt_future_states'][0]
        neighbor_future_states = data['gt_future_states'][1]
        object_type = data['object_type']

        return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type


class DrivingTestData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        scenario_id = self.data_list[idx].split('/')[-1].split('_')[0]

        ego = data['ego']
        neighbor = data['neighbors']
        map_lanes = data['map_lanes'][:, :, :200:2]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]

        object_type = data['object_type']
        current_state = data['current_state']
        object_index = data['object_index']

        return ego, neighbor, map_lanes, map_crosswalks, object_type, current_state, object_index, scenario_id


def gmm_loss(trajectories, convs, log_probs, ground_truth):
    metric_time = [29, 49, 79]
    distance = torch.norm(trajectories[:, :, :, : ,:2] - ground_truth[:, :, None, :, :2], dim=-1)
    ndistance = distance.mean(-1) + distance[..., metric_time].sum(-1) 
    best_mode = torch.argmin(ndistance.mean(1), dim=-1)
    B, N = trajectories.shape[0], trajectories.shape[1]
    
    #[B, N, T, 2]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], 
                                    best_mode[:, None, None]].squeeze(2)
    #[B, N, T, 2]
    convs = convs[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]].squeeze(2)

    dx = best_mode_future[...,0] - ground_truth[...,0]
    dy = best_mode_future[...,1] - ground_truth[...,1]

    log_std_x = torch.clamp(convs[..., 0], -2, 2)
    log_std_y = torch.clamp(convs[..., 1], -2, 2)
    std_x, std_y = torch.exp(log_std_x), torch.exp(log_std_y)

    reg_gmm_log_coefficient = log_std_x + log_std_y
    reg_gmm_exp = 0.5 * ((dx ** 2) / (std_x ** 2) + (dy ** 2) / (std_y ** 2))
    loss = reg_gmm_log_coefficient + reg_gmm_exp
    loss = loss.mean(-1) + loss[..., metric_time].sum(-1)

    prob_loss = F.cross_entropy(log_probs, best_mode, label_smoothing=0.2)
    loss = loss + prob_loss
    loss = loss.mean()

    return loss, best_mode, best_mode_future, convs


def level_k_loss(outputs, ego_future, neighbor_future):
    loss: torch.tensor = 0
    levels = len(outputs.keys()) // 2
    neighbor_future_valid = torch.ne(neighbor_future[..., :2].sum(-1), 0)
    ego_future_valid = torch.ne(ego_future[..., :2].sum(-1), 0)

    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions'][..., :2]
        scores = outputs[f'level_{k}_scores']
        ego = trajectories[:, 0] * ego_future_valid.unsqueeze(1).unsqueeze(-1)
        neighbor = trajectories[:, 1] * neighbor_future_valid.unsqueeze(1).unsqueeze(-1)
        trajectories = torch.stack([ego, neighbor], dim=1)
        gt_future = torch.stack([ego_future, neighbor_future], dim=1)
        convs = outputs[f'level_{k}_interactions'][..., 2:]
        gmmloss, best_mode, future, _ = gmm_loss(trajectories, convs, scores.sum(1), gt_future)
        loss += gmmloss

    return loss, (future, best_mode, scores)


def motion_metrics(trajectories, ego_future, neighbor_future):
    ego_future_valid = torch.ne(ego_future[..., :2], 0)
    ego_trajectory = trajectories[:, 0] * ego_future_valid
    distance = torch.norm(ego_trajectory[:, 4::5, :2] - ego_future[:, 4::5, :2], dim=-1)
    egoADE = torch.mean(distance)
    egoFDE = torch.mean(distance[:, -1])

    neigbhor_future_valid = torch.ne(neighbor_future[..., :2], 0)
    neighbor_trajectory = trajectories[:, 1] * neigbhor_future_valid
    distance = torch.norm(neighbor_trajectory[:, 4::5, :2] - neighbor_future[:, 4::5, :2], dim=-1)
    neighborADE = torch.mean(distance)
    neighborFDE = torch.mean(distance[:, -1])

    return egoADE.item(), egoFDE.item(), neighborADE.item(), neighborFDE.item()


# Define metrics to measure the prediction
def default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
        track_steps_per_second: 10
        prediction_steps_per_second: 2
        track_history_samples: 10
        track_future_samples: 80
        speed_lower_bound: 1.4
        speed_upper_bound: 11.0
        speed_scale_lower: 0.5
        speed_scale_upper: 1.0
        step_configurations {
            measurement_step: 5
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
        }
        step_configurations {
            measurement_step: 9
            lateral_miss_threshold: 1.8
            longitudinal_miss_threshold: 3.6
        }
        step_configurations {
            measurement_step: 15
            lateral_miss_threshold: 3.0
            longitudinal_miss_threshold: 6.0
        }
        max_predictions: 6
    """
    text_format.Parse(config_text, config)

    return config


class MotionMetrics:
    """Wrapper for motion metrics computation."""
    def __init__(self):
        super().__init__()
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._metrics_config = default_metrics_config()

    def reset_states(self):
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []

    def update_state(self, prediction_trajectory, prediction_score, ground_truth_trajectory, ground_truth_is_valid, object_type):
        self._prediction_trajectory.append(prediction_trajectory[..., 4::5,:].clone().detach().cpu())
        self._prediction_score.append(prediction_score.clone().detach().cpu())
        self._ground_truth_trajectory.append(ground_truth_trajectory.cpu())
        self._ground_truth_is_valid.append(ground_truth_is_valid[..., -1].cpu())
        self._object_type.append(object_type.cpu())

    def result(self):
        # [batch_size, 1, top_k, 2, steps, 2].
        prediction_trajectory = torch.cat(self._prediction_trajectory, dim=0)
        # [batch_size, 1, top_k].
        prediction_score = torch.cat(self._prediction_score, dim=0)
        # [batch_size, 1, 2, gt_steps, 7].
        ground_truth_trajectory = torch.cat(self._ground_truth_trajectory, dim=0)
        # [batch_size, 1, gt_steps].
        ground_truth_is_valid = torch.cat(self._ground_truth_is_valid, dim=0)
        # [batch_size, 1].
        object_type = torch.cat(self._object_type, dim=0)

        # We are predicting more steps than needed by the eval code. Subsample.
        #interval = (self._metrics_config.track_steps_per_second // self._metrics_config.prediction_steps_per_second)
        # Prepare these into shapes expected by the metrics computation.
        # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        # num_agents_per_joint_prediction is 1 here.
        if len(prediction_trajectory.shape)>=4:
            prediction_trajectory = prediction_trajectory.unsqueeze(dim=1).numpy()
            prediction_score = prediction_score.unsqueeze(dim=1).numpy()
        else:
            prediction_trajectory = prediction_trajectory.numpy()

        # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].

        ground_truth_trajectory = ground_truth_trajectory.numpy()
        # [batch_size, num_agents_per_joint_prediction, gt_steps].
        ground_truth_is_valid = ground_truth_is_valid.numpy()

        # [batch_size, num_agents_per_joint_prediction].
        object_type = object_type.numpy()
        b = ground_truth_trajectory.shape[0]

        prediction_ground_truth_indices = tf.cast(tf.concat([tf.zeros((b, 1, 1)), tf.ones((b, 1, 1))],axis=-1),tf.int64)
        # print(object_type.shape)
        ground_truth_is_valid = tf.convert_to_tensor(ground_truth_is_valid)
        prediction_ground_truth_indices_mask = tf.ones_like(prediction_ground_truth_indices, dtype=tf.float32)
        valid_gt_all = tf.cast(tf.math.greater_equal(tf.reduce_sum(tf.cast(ground_truth_is_valid,tf.float32), axis=-1), 1), tf.float32)
        valid_gt_all = valid_gt_all[:, tf.newaxis, :] * prediction_ground_truth_indices_mask
        valid_gt_all = tf.cast(valid_gt_all, tf.bool)

        metric_values = py_metrics_ops.motion_metrics(
                config=self._metrics_config.SerializeToString(),
                prediction_trajectory=tf.convert_to_tensor(prediction_trajectory),
                prediction_score=tf.convert_to_tensor(prediction_score),
                ground_truth_trajectory=tf.convert_to_tensor(ground_truth_trajectory),
                ground_truth_is_valid=ground_truth_is_valid,
                object_type=tf.convert_to_tensor(object_type),
                prediction_ground_truth_indices=prediction_ground_truth_indices,
                prediction_ground_truth_indices_mask=valid_gt_all)


        metric_names = config_util.get_breakdown_names_from_motion_config(self._metrics_config)
        results = {}

        for i, m in enumerate(['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']):
            for j, n in enumerate(metric_names):
                results[f'{m}_{n}'] = metric_values[i][j].numpy()

        return results
