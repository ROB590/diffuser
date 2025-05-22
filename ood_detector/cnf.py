import torch
import h5py
import math
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Flow Matching API imports:
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from torch.distributions import Independent, Normal

torch.manual_seed(42)

##############################################
# 1) Define the Flow Model
##############################################
class Flow(nn.Module):
    def __init__(self, dim, hidden_dim=2048):  #hidden_dim=600
        """
        Args:
            dim (int): Dimensionality of the input (observation + flattened actions).
            hidden_dim (int): Number of hidden units.
        """
        super(Flow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0], 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat((t, x), -1))


##############################################
# 2) CNF Dataset with Train/Validation Split
##############################################
class CNFDataset(Dataset):
    def __init__(self, file_path="t_sc.hdf5", predict_horizon=2, use_action=False, split="train", train_ratio=0.8):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            predict_horizon (int): Horizon for the actions.
            use_action (bool): Whether to use action predictions.
            split (str): Either "train" or "val". For "t_sc.hdf5", this splits the data.
            train_ratio (float): Fraction of data used for training when file_path == "t_sc.hdf5".
        """
        self.obs_action_pair = []
        self.use_action = use_action

        # Load all data from file.
        with h5py.File(file_path, 'r') as file:
            for ep in file:
                # print(file)
                # print(file.keys())
                # print(ep)
                episode = file[ep]
                #print(episode.keys())
                for each_ep in episode.keys():
                    step_keys = sorted(episode[each_ep].keys(), key=lambda s: int(s.split("_")[1]))
                    # print(step_keys)
                    for step in step_keys:
                        step_group = episode[each_ep][step]
                        # Use two timesteps (for point and state)
                        point = step_group["point"][()]  # e.g. shape (2, 64)
                        agent = step_group["state"][()]   # e.g. shape (2, 25)
                        
                        point_tensor = torch.from_numpy(point)   # (2, 64)
                        agent_tensor = torch.from_numpy(agent)     # (2, 25)
                        
                        # Concatenate across timesteps (flatten to one vector)
                        obs = torch.cat([point_tensor.flatten(), agent_tensor.flatten()])  # 2*64 + 2*25 = 178
                        if use_action:
                            actions = step_group['action_pred'][()]  
                            for i in range(len(actions) - predict_horizon):
                                self.obs_action_pair.append((
                                    obs, 
                                    torch.from_numpy(actions[i:i+predict_horizon])
                                ))
                        else:
                            actions = step_group['action_pred'][()]
                            for i in range(len(actions) - predict_horizon):
                                self.obs_action_pair.append((
                                    obs, 
                                    0  # use 0 as a placeholder
                                ))
            
        # If the file is t_sc.hdf5, perform a split into train/validation.
        if file_path == "t_sc.hdf5":
            total = len(self.obs_action_pair)
            split_idx = int(total * train_ratio)
            if split == "train":
                self.obs_action_pair = self.obs_action_pair[:split_idx]
            elif split == "val":
                self.obs_action_pair = self.obs_action_pair[split_idx:]
            else:
                raise ValueError("split must be either 'train' or 'val'")
        # For t_fc.hdf5, we use the entire data as a validation set.
    
    def __getitem__(self, idx):
        obs, action = self.obs_action_pair[idx]
        return obs, action
    
    def __len__(self):
        return len(self.obs_action_pair)
    
    def __dim__(self):
        if len(self.obs_action_pair) == 0:
            return 0
        obs_dim = self.obs_action_pair[0][0].shape  # e.g. (178,)
        if self.use_action:
            action_dim = self.obs_action_pair[0][1].shape
        else:
            action_dim = (0,0)
        print(f"<CNF Dataset> Obs dim: {obs_dim}, Action dim: {action_dim}")
        return obs_dim, action_dim


##############################################
# 3) Training with FlowMatching Solver
##############################################
def train(use_action=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    # Use only the training split from t_sc.hdf5
    dataset = CNFDataset(file_path="t_sc.hdf5", use_action=use_action, split="train", train_ratio=0.8)
    # Get observation dimension from dataset: expected 178 here.
    obs_dim, act_dim = dataset.__dim__()
    act_dim = act_dim[0] * act_dim[1] 
    # Initialize the CNF model with the proper input dimension.
    model = Flow(dim=obs_dim[0] + act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=8)
    # Create checkpoint directory if not exists.
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = AffineProbPath(scheduler=CondOTScheduler())

    num_epochs = 3000
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (obs, actions) in enumerate(pbar):
            optimizer.zero_grad()
            obs = obs.to(device)
            if use_action:
                actions = actions.to(device)
                actions_flat = actions.view(actions.shape[0], -1)
                x1 = torch.cat([obs, actions_flat], dim=-1)  # target sample
            else:
                x1 = obs  # target sample
            x0 = torch.randn_like(x1)  # random noise sample
            t = torch.rand(len(x1), device=device)
            # Debug print (can be commented out)
            print(t.shape, x0.shape, x1.shape)
            path_sample = path.sample(t=t, x_0=x0, x_1=x1)
            loss = torch.pow(model(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        if avg_loss < 0.01:
            print("Loss is less than 0.01, training is stopped")
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")
            break

        if (epoch + 1) % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

    return model


##############################################
# 4) Evaluation with ODESolver using compute_likelihood
##############################################
def evaluate_with_odesolver(model, dataset_path, use_action=False, num_acc=10, step_size=0.001, split=None):
    """
    This function evaluates the model on a dataset by computing the log-likelihood
    using the ODESolver's compute_likelihood method. For each batch, it averages over 
    'num_acc' runs using the Hutchinson estimator (approximate divergence) and also 
    computes the exact divergence version.
    
    Args:
        dataset_path (str): Path to the HDF5 file.
        split (str or None): If provided ("train" or "val"), split the data when dataset_path is "t_sc.hdf5".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on device:", device)
    
    wrapped_model = ModelWrapper(model)
    odesolver = ODESolver(velocity_model=wrapped_model)
    
    # When evaluating t_sc, use the split argument; for t_fc, split is ignored.
    dataset = CNFDataset(file_path=dataset_path, use_action=use_action, split=split if split else "train")
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=8)
    
    all_ll_est = []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {dataset_path}")
        for obs, action in pbar:
            obs = obs.to(device)
            if use_action:
                action = action.to(device)
                actions_flat = action.view(action.shape[0], -1)
                x = torch.cat([obs, actions_flat], dim=-1)
            else:
                x = obs
            # Define base density: isotropic Gaussian over the x dimension.
            d = x.shape[1]
            print("dimension for gaussian is ", d)
            gaussian_log_density = Independent(
                Normal(torch.zeros(d, device=device), torch.ones(d, device=device)),
                1
            ).log_prob
            
            # Compute log likelihood using the Hutchinson estimator
            log_p_acc = 0.0
            for i in range(num_acc):
                _, log_p = odesolver.compute_likelihood(
                    x_1=x,
                    method='midpoint',
                    step_size=step_size,
                    exact_divergence=False,
                    log_p0=gaussian_log_density
                )
                log_p_acc += log_p
            log_p_acc /= num_acc
            
            all_ll_est.append(log_p_acc.cpu())
            avg_ll_est = log_p_acc.mean().item()
            pbar.set_postfix(est=f"{avg_ll_est:.2f}")
    
    all_ll_est = torch.cat(all_ll_est, dim=0)
    mean_ll_est = all_ll_est.mean().item()
    return mean_ll_est


##############################################
# 5) Main: Training and Evaluation
##############################################
if __name__ == "__main__":
    # Optionally check dataset details
    full_dataset = CNFDataset(file_path="t_sc.hdf5")
    print("Full t_sc Dataset length:", len(full_dataset))
    print("Dataset dim:", full_dataset.__dim__())
    print("Dataset sample:", full_dataset[0])
    
    # Train the model using only 80% of the t_sc data.
    model = train()
    
    # Optionally reload a saved model checkpoint
    reload_flag = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reload_path = "checkpoints/model_epoch_3000.pth"
    if reload_flag:
        model = Flow(dim=178).to(device)
        model.load_state_dict(torch.load(reload_path))
    model.eval()
    
    # Evaluate on the validation split from t_sc (the remaining 20%)
    success_ll_est = evaluate_with_odesolver(model, dataset_path="t_sc.hdf5", split="val")
    # Evaluate on all of t_fc as validation data
    failure_ll_est = evaluate_with_odesolver(model, dataset_path="t_fc.hdf5")
    
    print("=== Evaluation Results ===")
    print("Success (In-distribution, validation split) Data:")
    print("  Estimated Likelihood:", success_ll_est)
    print("Failure (Out-of-distribution) Data:")
    print("  Estimated Likelihood:", failure_ll_est)