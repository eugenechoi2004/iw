import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.long)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

def generate_points_vectorized(num_points, trajectories, episode_lengths, min_L, max_L):
    episode_lengths = np.array([len(traj) for traj in trajectories])

    num_trajectories = len(trajectories)
    traj_indices = np.random.randint(0, num_trajectories, size=num_points)
    traj_lengths = episode_lengths[traj_indices]


    L = np.random.randint(min_L, max_L + 1, size=num_points)
    valid_mask = (L <= traj_lengths)
    while not valid_mask.all():
        invalid_indices = np.where(~valid_mask)[0]
        traj_indices[invalid_indices] = np.random.randint(0, num_trajectories, size=len(invalid_indices))
        traj_lengths[invalid_indices] = episode_lengths[traj_indices[invalid_indices]]
        L[invalid_indices] = np.random.randint(min_L, max_L + 1, size=len(invalid_indices))
        valid_mask = (L <= traj_lengths)

    t0 = np.random.randint(0, traj_lengths - L + 1)
    t2 = t0 + L - 1

    t1 = np.zeros_like(t0)
    odd_mask = (L % 2 == 1)
    even_mask = ~odd_mask
    t1[odd_mask] = t0[odd_mask] + L[odd_mask] // 2
    t1[even_mask] = t0[even_mask] + (L[even_mask] // 2) - 1

    starts = np.array([trajectories[traj_idx][t0_idx] for traj_idx, t0_idx in zip(traj_indices, t0)])
    waypoints = np.array([trajectories[traj_idx][t1_idx] for traj_idx, t1_idx in zip(traj_indices, t1)])
    goals = np.array([trajectories[traj_idx][t2_idx] for traj_idx, t2_idx in zip(traj_indices, t2)])

    return starts, waypoints, goals

def create_data_loader(starts, waypoints, goals, batch_size, shuffle=True):
    dataset = np.stack([starts, waypoints, goals], axis=1)
    trajectory_dataset = TrajectoryDataset(dataset)
    data_loader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
