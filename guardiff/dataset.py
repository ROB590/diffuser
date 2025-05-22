import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#Trajectory-embedding 
# Transition-prediction
    # label = 1.0 if 'succ.npz' in self.files[idx] else 0.0
    # sample['label'] = torch.tensor(label, dtype=torch.float32)
class Maze2DTrajectoryDataset(Dataset):
    """
    Loads .npz episodes saved by the recorder.
    If split='succ', only success episodes; if 'fail', only failures; if None, both.
    """
    def __init__(self, root_dir, split=None, transform=None):
        super().__init__()
        pattern = '*_succ.npz' if split=='succ' else '*_fail.npz' if split=='fail' else '*.npz'
        self.files = sorted(glob.glob(os.path.join(root_dir, pattern)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # shape: (T, dim)
        states     = data['state']       # (T,4)
        actions    = data['action']      # (T,2)
        next_states= data['next_state']  # (T,4)
        rewards    = data['reward']      # (T,)
        dones      = data['done']        # (T,)

        sample = {
            'state':      torch.from_numpy(states),
            'action':     torch.from_numpy(actions),
            'next_state': torch.from_numpy(next_states),
            'reward':     torch.from_numpy(rewards),
            'done':       torch.from_numpy(dones.astype(np.float32))
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

# Usage
train_succ = Maze2DTrajectoryDataset(data_root, split='succ')
train_fail = Maze2DTrajectoryDataset(data_root, split='fail')
both       = Maze2DTrajectoryDataset(data_root, split=None)

loader_succ = DataLoader(train_succ, batch_size=16, shuffle=True)
loader_fail = DataLoader(train_fail, batch_size=16, shuffle=True)