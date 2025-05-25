import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
# Flow-Matching / CNF imports
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from torch.distributions import Independent, Normal
from typing import Optional
torch.manual_seed(42)

# --- 1. Single-Step Dataset ----------------------------------------------
class SingleStepTrajectoryDataset(Dataset):
    """
    One sample per timestep: x1 = action_t,
    x0 = Gaussian noise, z = [state_t, goal].
    """
    def __init__(self, root_dir, split=None):
        pattern = (
            '*_succ.npz' if split=='succ' else
            '*_fail.npz' if split=='fail' else
            '*.npz'
        )
        files = sorted(glob.glob(os.path.join(root_dir, pattern)))
        self.samples = []
        for f in files:
            data    = np.load(f)
            states  = torch.from_numpy(data['state']).float()   # (T, S)
            actions = torch.from_numpy(data['action']).float()  # (T, A)
            goal    = torch.from_numpy(data['goal']).float()    # (G,)
            T, S    = states.shape
            _, A    = actions.shape
            G       = goal.numel()
            for t in range(T):
                x1 = actions[t]                              # (A,)
                x0 = torch.randn_like(x1)                    # (A,)
                z  = torch.cat([states[t], goal], dim=-1)    # (S+G,)
                self.samples.append({'x0': x0, 'x1': x1, 'z': z})
        if not self.samples:
            raise RuntimeError(f"No data under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def dims(self):
        s = self.samples[0]
        return s['x1'].numel(), s['z'].numel()


# --- 2A. Cross-Attention CNF Model (seq_len=1) ----------------------------
class ContextActionCrossAttention(nn.Module):
    def __init__(self, A_dim, Z_dim, hidden=256, num_heads=4, seq_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.A_step  = A_dim
        self.Z_step  = Z_dim
        # projections (each step only)
        self.q_proj = nn.Linear(self.A_step + 1, hidden)
        self.k_proj = nn.Linear(self.Z_step + 1, hidden)
        self.v_proj = nn.Linear(self.Z_step + 1, hidden)
        # positional embeddings (redundant when seq_len=1, but kept for API)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden))
        self.attn   = nn.MultiheadAttention(embed_dim=hidden,
                                           num_heads=num_heads,
                                           batch_first=True)
        # internal FFN (hidden→hidden)
        self.internal_ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.ELU(),
            nn.Linear(hidden * 4, hidden),
        )
        # read-out to A_dim
        self.readout = nn.Linear(hidden, self.A_step)

    def forward(self, x_flat, t, z_flat):
        N = x_flat.size(0)
        # reshape to [N,1,A_step] & [N,1,Z_step]
        x_seq = x_flat.view(N, self.seq_len, self.A_step)
        z_seq = z_flat.view(N, self.seq_len, self.Z_step)
        # time embedding
        t_vec = t.unsqueeze(-1) if t.dim()==1 else t  # [N,1]
        t_seq = t_vec.unsqueeze(1)                    # [N,1,1]
        # project + pos emb
        q = self.q_proj(torch.cat([x_seq, t_seq], -1)) + self.pos_emb
        k = self.k_proj(torch.cat([z_seq, t_seq], -1)) + self.pos_emb
        v = self.v_proj(torch.cat([z_seq, t_seq], -1)) + self.pos_emb
        # attend
        attn_out, _ = self.attn(q, k, v)              # [N,1,hidden]
        h_int       = attn_out + self.internal_ffn(attn_out)
        out_seq     = self.readout(h_int)             # [N,1,A_step]
        return out_seq.view(N, -1)                    # [N,A_dim]


class ConditionalFlowAttention(nn.Module):
    def __init__(self, A_dim, Z_dim, hidden=256, num_heads=4):
        super().__init__()
        self.net = ContextActionCrossAttention(A_dim, Z_dim,
                                               hidden, num_heads,
                                               seq_len=1)
    def forward(self, x, t, z):
        return self.net(x, t, z)


# --- 2B. Optional: Simple MLP CNF Model ----------------------------------
class ConditionalFlowMLP(nn.Module):
    """
    f(x,t,z) -> dx/dt via MLP.
    """
    def __init__(self, A_dim, Z_dim, hidden=512, n_layers=3):
        super().__init__()
        layers = []
        in_dim = A_dim + Z_dim + 1
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ELU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, A_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, z):
        t_vec = t.unsqueeze(-1) if t.dim()==1 else t
        inp   = torch.cat([x, z, t_vec], dim=-1)
        return self.net(inp)  # [N, A_dim]


# --- 3. Velocity Wrappers for ODESolver -------------------------------
class ConditionalVelocityWrapper(nn.Module):
    def __init__(self, cnf_net, z_flat):
        super().__init__()
        self.cnf    = cnf_net
        self.z_flat = z_flat

    def forward(self, t, x):
        # t: scalar or [N], x: [N,A]
        batch_z = self.z_flat.expand(x.size(0), -1)
        return self.cnf(x, t.expand(x.size(0)), batch_z)


# --- 4. Training Function ----------------------------------------------
def train_single_step_cnf(
    traj_root, epochs=1000, batch_size=4096, lr=2e-4,
    num_workers=8, device=None, ckpt_dir="./ckpts"
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)

    ds     = SingleStepTrajectoryDataset(traj_root, split='succ')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    A_dim, Z_dim = ds.dims()
    # model = ConditionalFlowAttention(A_dim, Z_dim).to(device) #FIXME may contain bug
    model = ConditionalFlowMLP(A_dim, Z_dim).to(device)

    try:
        model = torch.compile(model)
        print("✅ Model compiled with torch.compile")
    except:
        pass

    optimizer = optim.Adam(model.parameters(), lr=lr)
    path      = AffineProbPath(scheduler=CondOTScheduler())

    for ep in range(1, epochs+1):
        total_loss = 0.0
        model.train()
        for batch in loader:
            x0 = batch['x0'].to(device)      # noise
            x1 = batch['x1'].to(device)      # true action
            z  = batch['z'].to(device)       # context
            N  = x1.size(0)
            t  = torch.rand(N, device=device)
            sample = path.sample(t=t, x_0=x0, x_1=x1)
            v_pred = model(sample.x_t, sample.t, z)  # [N, A_dim]
            loss   = (v_pred - sample.dx_t).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {ep}/{epochs}, Loss: {avg:.6f}")
        if ep % 200 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ep{ep}.pth"))


# --- 5. Evaluation Function --------------------------------------------
def eval_trajectory_ll(
    model: nn.Module, traj_path: str,
    num_acc: int = 10, step_size: float = 0.01,
    device: Optional[torch.device] = None
) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    data    = np.load(traj_path)
    states  = torch.from_numpy(data['state']).float()
    actions = torch.from_numpy(data['action']).float()
    goal    = torch.from_numpy(data['goal']).float()
    T, S    = states.shape
    _, A    = actions.shape
    G       = goal.numel()

    # per-step batching
    x1_list, z_list = [], []
    for t in range(T):
        x1_list.append(actions[t])
        z_list.append(torch.cat([states[t], goal], dim=-1))

    x1_batch = torch.stack(x1_list, dim=0).to(device)  # (T, A)
    z_batch  = torch.stack(z_list,  dim=0).to(device)  # (T, S+G)
    x0_batch = torch.randn_like(x1_batch)

    base_dist = Independent(
        Normal(torch.zeros(A, device=device),
               torch.ones(A,  device=device)),
        1
    ).log_prob

    wrapper = ConditionalVelocityWrapper(model, z_batch)
    solver  = ODESolver(velocity_model=wrapper)

    ll_acc = torch.zeros(T, device=device)
    for _ in trange(num_acc, desc="Accumulate LL"):
        with torch.no_grad():
            _, ll = solver.compute_likelihood(
                x_1              = x1_batch,
                method           = 'midpoint',
                step_size        = step_size,
                exact_divergence = True,
                log_p0           = base_dist
            )
        ll_acc += ll

    return (ll_acc / num_acc).cpu().numpy()


# --- 6. Main Entrypoint -----------------------------------------------
if __name__ == "__main__":
    ROOT   = "/home/haoran-zhang/Desktop/projects/guardiff/trajectories"
    CKPTS  = "./checkpoints/cnf_single"
    EPOCHS = 1000

    # 1) Train
    train_single_step_cnf(ROOT, epochs=EPOCHS, ckpt_dir=CKPTS)

    # 2) Load trained model
    ds    = SingleStepTrajectoryDataset(ROOT, split='succ')
    A_dim, Z_dim = ds.dims()
    net   = ConditionalFlowAttention(A_dim, Z_dim)
    ckpt  = os.path.join(CKPTS, f"ep{EPOCHS}.pth")
    net.load_state_dict(torch.load(ckpt))
    net.eval()

    # 3) Evaluate failed trajectories
    for fn in sorted(glob.glob(os.path.join(ROOT, "*_fail.npz"))):
        ll_vec = eval_trajectory_ll(net, fn)
        plt.figure(figsize=(8,3))
        plt.plot(ll_vec, marker='o')
        plt.title(f"LL scan for {os.path.basename(fn)}")
        plt.xlabel("Timestep")
        plt.ylabel("Avg log‐likelihood")
        plt.grid(True)
        plt.tight_layout()
        plt.show()