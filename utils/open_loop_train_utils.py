import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F


class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states
    

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def interaction_loss(trajectories, last_trajectories, neighbors_valid):
    B, N, M, T, _ = trajectories.shape
    neighbors_to_ego = []
    neighbors_to_neighbors = []
    neighbors_mask = neighbors_valid.logical_not()
    mask = torch.zeros(B, N-1, N-1).to(neighbors_mask.device)
    mask = torch.masked_fill(mask, neighbors_mask[:, :, None], 1)
    mask = torch.masked_fill(mask, neighbors_mask[:, None, :], 1)
    mask = torch.masked_fill(mask, torch.eye(N-1)[None, :, :].bool().to(neighbors_mask.device), 1)
    mask = mask.unsqueeze(-1).unsqueeze(-1) * torch.ones(1, 1, M, M).to(neighbors_mask.device)
    mask = mask.permute(0, 1, 3, 2, 4).reshape(B, (N-1)*M, (N-1)*M)

    for t in range(T):
        # AV-agents last level
        ego_p = trajectories[:, 0, :, t, :2]
        last_neighbors_p = last_trajectories[:, 1:, :, t, :2]
        last_neighbors_p = last_neighbors_p.reshape(B, -1, 2)
        dist_to_ego = torch.cdist(ego_p, last_neighbors_p)
        n_mask = neighbors_mask.unsqueeze(-1).expand(-1, -1, M).reshape(B, 1, -1)
        dist_to_ego = torch.masked_fill(dist_to_ego, n_mask, 1000)
        neighbors_to_ego.append(torch.min(dist_to_ego, dim=-1).values)

        # agents-agents last level
        neighbors_p = trajectories[:, 1:, :, t, :2].reshape(B, -1, 2)
        dist_neighbors = torch.cdist(neighbors_p, last_neighbors_p)
        dist_neighbors = torch.masked_fill(dist_neighbors, mask.bool(), 1000)
        neighbors_to_neighbors.append(torch.min(dist_neighbors, dim=-1).values)

    neighbors_to_ego = torch.stack(neighbors_to_ego, dim=-1)
    PF_to_ego = 1.0 / (neighbors_to_ego + 1)
    PF_to_ego = PF_to_ego * (neighbors_to_ego < 3) # safety threshold
    PF_to_ego = PF_to_ego.sum(-1).sum(-1).mean()
        
    neighbors_to_neighbors = torch.stack(neighbors_to_neighbors, dim=-1)
    PF_to_neighbors = 1.0 / (neighbors_to_neighbors + 1)
    PF_to_neighbors = PF_to_neighbors * (neighbors_to_neighbors < 3) # safety threshold
    PF_to_neighbors = PF_to_neighbors.sum(-1).mean(-1).mean() 

    return PF_to_ego + PF_to_neighbors


def imitation_loss(gmm, scores, ground_truth):
    distance = torch.norm(gmm[:, :, :, 4::5, :2] - ground_truth[:, :, None, 4::5, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1).sum(1), dim=-1)
    B, N = gmm.shape[0], gmm.shape[1]

    mu = gmm[..., :2]
    best_mode_mu = mu[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]]
    best_mode_mu = best_mode_mu.squeeze(2)
    dx = ground_truth[..., 0] - best_mode_mu[..., 0]
    dy = ground_truth[..., 1] - best_mode_mu[..., 1]
    
    cov = gmm[..., 2:]
    best_mode_cov = cov[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]]
    best_mode_cov = best_mode_cov.squeeze(2)
    log_std_x = torch.clamp(best_mode_cov[..., 0], -2, 2)
    log_std_y = torch.clamp(best_mode_cov[..., 1], -2, 2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
    loss = torch.mean(gmm_loss) + torch.mean(gmm_loss[:, 0])

    best_mode = best_mode.unsqueeze(1).expand(-1, N)
    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, label_smoothing=0.2)
    loss += score_loss

    return loss, best_mode_mu


def level_k_loss(outputs, ego_future, neighbors_future, neighbors_future_valid):
    loss: torch.tensor = 0
    levels = len(outputs.keys()) // 2

    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions']
        scores = outputs[f'level_{k}_scores']
        predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
        plan = trajectories[:, :1]
        trajectories = torch.cat([plan, predictions], dim=1)
        gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)
        il_loss, future = imitation_loss(trajectories, scores, gt_future)
        neighbors_valid = neighbors_future_valid[:, :, 0, 0]
        loss += il_loss
        if k >= 1:
            inter_loss = interaction_loss(outputs[f'level_{k}_interactions'], outputs[f'level_{k-1}_interactions'], neighbors_valid)
            loss += 0.1 * inter_loss

    return loss, future


def motion_metrics(plan_trajectory, prediction_trajectories, ego_future, neighbors_future, neighbors_future_valid):
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()


def speed_comfort_loss(trajectories):
    a, j, y = inverse_dynamic_model(trajectories)
    loss = 0.01 * torch.mean(-a * (a < 0)) 
    loss += 0.001 * torch.mean(j.abs())
    loss += 0.001 * torch.mean(y.abs())

    return loss


def inverse_dynamic_model(trajs):
    dt = 0.1
    dp = torch.diff(trajs, dim=-2)
    v = torch.norm(dp, dim=-1) / dt
    theta = torch.atan2(dp[..., 1], dp[..., 0].clamp(min=1e-3))
    d_v = torch.diff(v)
    a = d_v / dt
    d_theta = torch.diff(theta)
    y = d_theta / dt
    d_a = torch.diff(a)
    j = d_a / dt

    return a, j, y
