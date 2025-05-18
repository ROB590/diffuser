# # They have sort of rollout Augmentation [question1]
# randomly add or drop frames
# Probe architecture [original paper]
# Probe architecture [modified input as activation layer]
#NOTE more window length more accurate both valid and train loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm

# Dictionaries for mapping string labels to tensors
label_dict = {
    'grasp': torch.tensor([0], dtype=torch.long),
    'plan': torch.tensor([1], dtype=torch.long),
}
state_dict = {
    'success': torch.tensor([0], dtype=torch.long),
    'fail': torch.tensor([1], dtype=torch.long),
}

# -----------------------------
# 1. Dataset - Modified to use past window to predict current state
# -----------------------------
class ProbeTrajectoryDataset(Dataset):
    def __init__(self, hdf5_path, window_size=2, use_lang=True):
        """
        hdf5_path: path to the HDF5 file storing episodes.
        window_size: number of previous steps to use for predicting current state.
        use_lang: whether to use the language/task embedding.
        
        The dataset will return (previous window, task embedding, current label).
        """
        self.window_size = window_size
        self.use_lang = use_lang
        self.data = []  # List to hold all preprocessed samples
        
        # Load and process all data from the HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            for ep in f:
                episode = f[ep]
                for ep_key in episode.keys():
                    # Get sorted step keys (assuming naming like "step_0", "step_1", etc.)
                    step_keys = sorted(episode[ep_key].keys(), key=lambda s: int(s.split("_")[1]))
                    
                    f_h_list = []    # List to store per-step history features for this episode
                    label_list = []  # List to store per-step labels
                    f_z = None       # Task/language embedding (assumed consistent across steps)
                    
                    # Process each step in the episode
                    for step in step_keys:
                        group = episode[ep_key][step]
                        
                        # Extract modalities for this step
                        point = group["point"][()]
                        agent = group["state"][()]
                        action = group["action"][()]
                        action_pred = group["action_pred"][()]
                        
                        # Convert to torch tensors
                        point_tensor = torch.from_numpy(point)
                        agent_tensor = torch.from_numpy(agent)
                        action_tensor = torch.from_numpy(action)
                        action_pred_tensor = torch.from_numpy(action_pred)
                        
                        # Concatenate flattened features to create the per-step history feature
                        f_h_step = torch.cat([
                            point_tensor.flatten(), 
                            agent_tensor.flatten(), 
                            action_tensor.flatten(), 
                            action_pred_tensor.flatten()
                        ])
                        f_h_list.append(f_h_step)  # Shape: [feature_dim]
                        
                        # Process label for this step
                        label = group["success_label"][()]
                        label = label.decode('utf-8')
                        label_tensor = state_dict[label]  # Convert label string to tensor via state_dict
                        label_list.append(label_tensor)
                        
                        # Extract language/task embedding only once per episode
                        if self.use_lang and f_z is None:
                            if 'lang' in group:
                                f_z_np = group['lang'][()]
                            elif 'task_desc' in group:
                                f_z_np = group['task_desc'][()]
                            else:
                                raise ValueError("No task descriptor found in episode group.")
                            f_z = torch.tensor(f_z_np, dtype=torch.float32)
                    
                    # If language/task embedding is not used or not found, use a default zero vector.
                    if f_z is None:
                        f_z = torch.zeros(48, dtype=torch.float32)
                    
                    # Convert to tensors
                    f_h_all = torch.stack(f_h_list)    # Shape: [T, feature_dim]
                    labels_all = torch.stack(label_list) # Shape: [T]
                    num_steps = f_h_all.shape[0]
                    
                    # Create samples: use past window to predict current state
                    if num_steps > self.window_size:
                        for current_idx in range(self.window_size, num_steps):
                            # Get window of previous steps
                            f_h_window = f_h_all[current_idx - self.window_size:current_idx]
                            # Get current step's label (a single label per window)
                            current_label = labels_all[current_idx]
                            # Save this sample as a tuple: (history window, task embedding, current label)
                            self.data.append((f_h_window, f_z, current_label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Return preprocessed sample from memory
        return self.data[index]
    
    def __dim__(self):
        fh_dim = self.data[0][0].shape[1]  # shape: [window_size, dim]
        fz_dim = self.data[0][1].shape[0]  # shape: [dim]
        print(f"<Rollout Dataset> fh dim: {fh_dim}") 
        print(f"<Rollout Dataset> fz dim: {fz_dim}")
        return fh_dim, fz_dim

# -----------------------------
# 2. PrObe Model - Modified to predict single label from window
# -----------------------------
class PatternExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gate_layer = nn.Linear(embedding_dim, embedding_dim)
        self.inst_norm = nn.InstanceNorm1d(embedding_dim, affine=False)

    def forward(self, f_h):
        norm = torch.norm(f_h, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        f_unit = f_h / norm
        gate = torch.sigmoid(self.gate_layer(f_unit))
        sign_f = torch.sign(f_unit)
        f_p = sign_f * gate
        f_p = f_p.permute(0, 2, 1)
        f_p = self.inst_norm(f_p)
        f_p = f_p.permute(0, 2, 1)
        return f_p

class ProbeModel(nn.Module):
    def __init__(self, fh_dim=128, fz_dim=512, hidden_dim=256):
        super().__init__()
        self.pattern_extractor = PatternExtractor(fh_dim)
        self.lstm = nn.LSTM(input_size=fh_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,dropout=0.1)
        self.task_inst_norm = nn.LayerNorm(fz_dim)
        self.task_linear = nn.Linear(fz_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, f_h, f_z):
        f_p = self.pattern_extractor(f_h)  # [B, T, embedding_dim]
        lstm_out, (h_n, _) = self.lstm(f_p)  # lstm_out: [B, T, hidden_dim]
        f_flow = lstm_out
        final_h = h_n.squeeze(0)  # [B, hidden_dim]
        f_z_norm = self.task_inst_norm(f_z)
        f_z_trans = torch.tanh(self.task_linear(f_z_norm))  # [B, hidden_dim]
        fused = torch.cat([final_h, f_z_trans], dim=-1)  # [B, hidden_dim*2]
        logits = self.classifier(fused)  # [B, 1]
        return logits, f_p, f_flow

# -----------------------------
# 3. Loss Function with Weighting
# -----------------------------
def bce_loss_cls(logits, labels, pos_weight):
    """
    Classification loss using binary cross-entropy with weighting.
    logits: [B, 1]
    labels: [B] or [B, 1]
    pos_weight: Tensor specifying weight for positive (fail) examples.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if labels.dim() < logits.dim():
        labels = labels.unsqueeze(-1)  # [B] -> [B, 1]
    return criterion(logits, labels.float())

def total_loss_fn(logits, f_p, labels, pos_weight):
    return bce_loss_cls(logits, labels, pos_weight)

# -----------------------------
# 4. Training Function
# -----------------------------
def train_probe(model, train_loader, valid_loader, optimizer, device, pos_weight, num_epochs=1000, 
                patience=10, min_delta=0.001, checkpoint_dir="aed_checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        t_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in t_pbar:
            f_h, f_z, labels = batch
            f_h = f_h.to(device)
            f_z = f_z.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, f_p, f_flow = model(f_h, f_z)
            loss = total_loss_fn(logits, f_p, labels, pos_weight)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).long()
            if labels.dim() < pred.dim():
                labels = labels.unsqueeze(-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            t_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total * 100
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                f_h, f_z, labels = batch
                f_h = f_h.to(device)
                f_z = f_z.to(device)
                labels = labels.to(device)
                logits, f_p, f_flow = model(f_h, f_z)
                loss = total_loss_fn(logits, f_p, labels, pos_weight)
                val_loss += loss.item()
                pred = (torch.sigmoid(logits) > 0.5).long()
                if labels.dim() < pred.dim():
                    labels = labels.unsqueeze(-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(valid_loader)
        val_acc = correct / total * 100
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Final checkpoint saved at: {checkpoint_path}")
                break
        
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")
    
    return model, history

# -----------------------------
# 5. Main Function
# -----------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    batch_size = 4096
    hdf5_path_failure = "t_fc.hdf5"
    hdf5_path_success = "t_sc.hdf5"
    
    # Load datasets separately for failure and success
    dataset_failure = ProbeTrajectoryDataset(hdf5_path_failure, use_lang=False)
    dataset_success = ProbeTrajectoryDataset(hdf5_path_success, use_lang=False)
    
    print(f"Failure dataset samples: {len(dataset_failure)}")
    print(f"Success dataset samples: {len(dataset_success)}")
    
    # Compute pos_weight based on the counts
    # Since in state_dict, 'fail' is encoded as 1 and 'success' as 0,
    # pos_weight = (# success) / (# failure)
    num_failure = len(dataset_failure)
    num_success = len(dataset_success)
    pos_weight_value = num_success / num_failure
    pos_weight = torch.tensor(pos_weight_value).to(device)
    print(f"Computed pos_weight: {pos_weight.item():.4f}")
    
    # Combine datasets
    dataset = ConcatDataset([dataset_failure, dataset_success])
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Total trajectories: {len(dataset)}")
    print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
    
    # Initialize model and optimizer
    fh_dim, fz_dim = dataset_failure.__dim__()  # assume dimensions are the same for both datasets
    model = ProbeModel(fh_dim=fh_dim, fz_dim=fz_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model, passing the computed pos_weight into the loss
    trained_model, history = train_probe(model, train_loader, val_loader, optimizer, device, pos_weight, num_epochs=1000, patience=20)