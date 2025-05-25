import os
import glob
import torch
import math
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
# Flow-Matching / CNF imports
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from tqdm import trange
from torch.distributions import Independent, Normal

torch.manual_seed(42)

# --- 1. Dataset with Flattened Sequence Windows ---------------------------
class ConditionalTrajectoryDataset(Dataset):
    """
    Preloads trajectories and precomputes sliding windows of length seq_len,
    flattens each window into a single vector for x1 and z:
      x1: (seq_len*(S+A),) concatenated [state, action] per timestep
      x0: same shape, sampled from N(0,I)
      z : (seq_len*(S+G),) concatenated [state, goal] per timestep
    """
    def __init__(self, root_dir, split=None, seq_len=10):
        pattern = '*_succ.npz' if split=='succ' else '*_fail.npz' if split=='fail' else '*.npz'
        files = sorted(glob.glob(os.path.join(root_dir, pattern)))
        self.samples = []
        self.seq_len = seq_len
        for f in files:
            data = np.load(f)
            s_full = torch.from_numpy(data['state']).float()  # (T, S)
            a_full = torch.from_numpy(data['action']).float() # (T, A)
            g = torch.from_numpy(data['goal']).float()        # (G,)
            T, S = s_full.shape
            _, A = a_full.shape
            G = g.numel()
            if T < seq_len:
                continue
            # precompute goal sequence
            g_seq = g.unsqueeze(0).expand(seq_len, -1)       # (seq_len, G)
            for t0 in range(T - seq_len + 1):
                s = s_full[t0:t0+seq_len]                    # (seq_len, S)
                a = a_full[t0:t0+seq_len]                    # (seq_len, A)
                # x1: [s, a] flatten
                x1_seq = torch.cat([a], dim=-1)            # (seq_len, S+A)
                x1 = x1_seq.reshape(-1)                       # (seq_len*(S+A),)
                # x0: base noise
                x0 = torch.randn_like(x1)
                # z: [s, g_seq] flatten
                z_seq = torch.cat([s, g_seq], dim=-1)        # (seq_len, S+G)
                z = z_seq.reshape(-1)                         # (seq_len*(S+G),)
                self.samples.append({'x0': x0, 'x1': x1, 'z': z})
        if not self.samples:
            raise RuntimeError(f"No valid windows of length {seq_len} in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def dims(self):
        sample = self.samples[0]
        return sample['x1'].shape[0], sample['z'].shape[0]

# --- 2. Cross-Attention Flow Model ------------------------------------------
class ContextActionCrossAttention(nn.Module):
    def __init__(self, A_dim, Z_dim, hidden=256, num_heads=4, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.A_step = A_dim // seq_len
        self.Z_step = Z_dim // seq_len
        # projections
        self.q_proj = nn.Linear(self.A_step + 1, hidden)
        self.k_proj = nn.Linear(self.Z_step + 1, hidden)
        self.v_proj = nn.Linear(self.Z_step + 1, hidden)
        # positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden))
        self.attn   = nn.MultiheadAttention(embed_dim=hidden,
                                           num_heads=num_heads,
                                           batch_first=True)
        # 1) Internal ffn: maps hidden → hidden
        self.internal_ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.ELU(),
            nn.Linear(hidden * 4, hidden),
        )
        # 2) Read-out head: maps hidden → per-step output
        self.readout = nn.Linear(hidden, self.A_step)

    def forward(self, x_flat, t, z_flat):
        N = x_flat.size(0)
        # reshape to sequences
        x_seq = x_flat.view(N, self.seq_len, self.A_step)
        z_seq = z_flat.view(N, self.seq_len, self.Z_step)
        # time embedding sequence
        if t.dim() == 1: t = t.unsqueeze(-1)
        t_seq = t.unsqueeze(1).expand(N, self.seq_len, 1)
        q = self.q_proj(torch.cat([x_seq, t_seq], -1)) + self.pos_emb
        k = self.k_proj(torch.cat([z_seq, t_seq], -1)) + self.pos_emb
        v = self.v_proj(torch.cat([z_seq, t_seq], -1)) + self.pos_emb

        attn_out, _ = self.attn(q, k, v)            # [N, seq_len, hidden]
        h_int = attn_out + self.internal_ffn(attn_out)
        # now project to dx‐space
        out_seq = self.readout(h_int)               # [N, seq_len, A_step]
        return out_seq.reshape(N, -1)               # [N, A_dim], matches dx_t

class ConditionalFlowAttention(nn.Module):
    """CNF model using cross attention with positional embeddings."""
    def __init__(self, A_dim, Z_dim, hidden=256, num_heads=4, seq_len=10):
        super().__init__()
        self.net = ContextActionCrossAttention(A_dim, Z_dim,
                                               hidden, num_heads,
                                               seq_len)
    def forward(self, x, t, z):
        return self.net(x, t, z)

class ConditionalVelocityWrapper(nn.Module):
    """Wrap CNF net into velocity model, binding context z_flat."""
    def __init__(self, cnf_net, z_flat):
        super().__init__()
        self.cnf    = cnf_net
        self.z_flat = z_flat
    def forward(self, t, x):
        z = self.z_flat.expand(x.size(0), -1)
        return self.cnf(x, t.expand(x.size(0),1), z)
class ConditionalFlowMLP(nn.Module):
    """
    Simple MLP f_θ(x, t, z) → dx/dt
    """
    def __init__(self, A_dim, Z_dim, hidden=512, n_layers=3):
        super().__init__()
        layers = []
        in_dim = A_dim + Z_dim + 1
        # Hidden MLP blocks
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ELU())
            in_dim = hidden
        # Final projection back to action‐dim
        layers.append(nn.Linear(hidden, A_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, z):
        # x: (N, A), t: (N,) or (N,1), z: (N, Z)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, z, t], dim=-1)
        return self.net(inp)


class ConditionalVelocityWrapperMLP(nn.Module):
    """
    Wraps the MLP into a velocity model for the ODESolver,
    binding a single context batch z.
    """
    def __init__(self, mlp_net, z):
        super().__init__()
        self.cnf = mlp_net
        self.z   = z

    def forward(self, t, x):
        # t: scalar or (N,), x: (N,A)
        batch_z = self.z.expand(x.size(0), -1)
        return self.cnf(x, t.expand(x.size(0),1), batch_z)

# --- 3. Training w/ torch.compile and optimized DataLoader -----------------
def train_conditional_cnf(
    traj_root, seq_len=10, epochs=1000,
    batch_size=4096, lr=2e-4, num_workers=8,
    device=None, ckpt_dir="./checkpoints"
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    ds = ConditionalTrajectoryDataset(traj_root, split='succ', seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=True)
    A_dim, Z_dim = ds.dims()
    model = ConditionalFlowAttention(A_dim, Z_dim).to(device)
    # model = ConditionalFlowMLP(A_dim, Z_dim, hidden=512, n_layers=3).to(device)
    try:
        model = torch.compile(model)
        print("Model compiled")
    except Exception:
        pass
    optimizer = optim.Adam(model.parameters(), lr=lr)
    path = AffineProbPath(scheduler=CondOTScheduler())

    for ep in range(1, epochs+1):
        total_loss = 0.0
        model.train()
        for batch in tqdm(loader, desc=f"Epoch {ep}/{epochs}"):
            x0 = batch['x0'].to(device)
            x1 = batch['x1'].to(device)
            z  = batch['z'].to(device)
            N = x1.size(0)
            t = torch.rand(N, device=device)
            sample = path.sample(t=t, x_0=x0, x_1=x1)
            v_pred = model(sample.x_t, sample.t, z)
            loss = (v_pred - sample.dx_t).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"→ Ep {ep}, avg loss {total_loss/len(loader):.4f}")
        if ep % 200 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch{ep}.pth"))

def eval_trajectory_ll(
    model: nn.Module,
    traj_path: str,
    seq_len: int = 10,
    num_acc: int = 10,
    step_size: float = 0.01,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Computes average log-likelihoods for every non-overlapping window of length `seq_len`
    in the trajectory at `traj_path`, doing one big batched ODE solve.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # load trajectory
    data    = np.load(traj_path)
    states  = torch.from_numpy(data['state']).float()   # (T, S)
    actions = torch.from_numpy(data['action']).float()  # (T, A)
    goal    = torch.from_numpy(data['goal']).float()    # (G,)
    T, S    = states.shape
    _, A    = actions.shape

    # compute window starts (non-overlapping)
    starts    = list(range(0, T - seq_len + 1, seq_len))
    n_windows = len(starts)

    # stack all windows into big batches
    x1_list, z_list = [], []
    g_seq = goal.unsqueeze(0).expand(seq_len, -1)       # (seq_len, G)

    for i in starts:
        s_win = states[i:i+seq_len]                     # (seq_len, S)
        a_win = actions[i:i+seq_len]                    # (seq_len, A)
        # x1 = [s, a] flattened
        x1_list.append(torch.cat([a_win], dim=-1).reshape(-1))
        # z  = [s, g_seq] flattened
        z_list .append(torch.cat([s_win, g_seq], dim=-1).reshape(-1))

    x1_batch = torch.stack(x1_list, dim=0).to(device)   # (n_windows, window_dim)
    z_batch  = torch.stack(z_list,  dim=0).to(device)   # (n_windows, context_dim)
    x0_batch = torch.randn_like(x1_batch)               # same shape

    window_dim = x1_batch.size(1)
    base_dist  = Independent(
        Normal(torch.zeros(window_dim, device=device),
               torch.ones(window_dim,  device=device)),
        1
    ).log_prob

    # wrap + solver
    wrapper = ConditionalVelocityWrapper(model, z_batch)
    solver  = ODESolver(velocity_model=wrapper)

    ll_acc = torch.zeros(n_windows, device=device)
    for _ in trange(num_acc, desc="Accumulating estimates", unit="run"):
        with torch.no_grad():
            _, ll = solver.compute_likelihood(
                x_1               = x1_batch,
                method            = 'rk4',
                step_size         = step_size,
                exact_divergence  = True,
                log_p0            = base_dist
            )
        ll_acc += ll

    # average and move back to CPU
    ll_vec = (ll_acc / num_acc).cpu().numpy()
    return ll_vec
# --- 5. Main ---------------------------------------------------------------
if __name__ == "__main__":
    ROOT     = "/home/haoran-zhang/Desktop/projects/guardiff/trajectories"
    SEQ_LEN  = 10
    # CKPT     = "./checkpoints/cnf_multistep/mlp_checkpoints"
    CKPT     = "./checkpoints/cnf_multistep/cross_attention_checkpoints"
    EPOCHS   = 1000
    # train_conditional_cnf(ROOT, seq_len=SEQ_LEN, epochs=EPOCHS, ckpt_dir=CKPT)
    # load model
    dataset = ConditionalTrajectoryDataset(ROOT, split='succ', seq_len=SEQ_LEN)
    A_dim, Z_dim = dataset.dims()
    net =  ConditionalFlowAttention(A_dim, Z_dim)
    raw_state = torch.load(os.path.join(CKPT, f"epoch{EPOCHS}.pth"))
    clean_state = {k.replace("_orig_mod.", ""): v for k,v in raw_state.items()}
    net.load_state_dict(clean_state)
    net.eval()

    # process each trajectory file
    all_files = sorted(glob.glob(os.path.join(ROOT, "*_fail.npz")))
    for traj_file in all_files:
        ll_vec = eval_trajectory_ll(net, traj_file, seq_len=SEQ_LEN)
        plt.figure(figsize=(8,3))
        plt.plot(ll_vec, marker='o')
        plt.title(f"LL trajectory scan for {os.path.basename(traj_file)}")
        plt.xlabel("Window start index")
        plt.ylabel("Average log-likelihood")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
