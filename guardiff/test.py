import os
import json
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.models import TemporalUnet, GaussianDiffusion
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch
import torch
from environment import maze2d
import time
# flow matching
from ood_detector.cnf_state_goal_cond_multistep import (
    ConditionalFlowAttention,
    ConditionalFlowMLP,
    ConditionalVelocityWrapper,
)
from flow_matching.solver import ODESolver
from torch.distributions import Independent, Normal
#######################
# Helper functions
#######################
seed = maze2d.ensure_seed()
def load_diffusion_manual(logbase, dataset_name, horizon, n_steps, epoch='latest', device='cuda'):
    base = os.path.join(logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')
    cfg_names = ['dataset', 'render', 'model', 'diffusion', 'trainer']
    cfgs = {}
    for name in cfg_names:
        path = os.path.join(base, f'{name}_config.pkl')
        cfgs[name] = pickle.load(open(path, 'rb'))

    dataset_obj = cfgs['dataset']()
    renderer    = cfgs['render']()
    model       = cfgs['model']().to(device)
    diffusion   = cfgs['diffusion'](model).to(device)
    trainer     = cfgs['trainer'](diffusion, dataset_obj, renderer)

    if epoch == 'latest':
        #epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')) #
        #FIXME hard code checkpoints
        epoch = 980000 #200000
        print("Current horizon",horizon)
        print('current step',n_steps)
    trainer.load(epoch)

    return DiffusionExperiment(
        dataset_obj, renderer, model, diffusion, trainer.ema_model, trainer, epoch
    )
# Flow matching helper
def init_ood_detector(model_type, A_dim, Z_dim, ckpt_path, device):
    """
    model_type: 'attention' or 'mlp'
    Loads weights from ckpt_path into the correct model, returns model.eval().
    """
    if model_type == 'attention':
        model = ConditionalFlowAttention(A_dim, Z_dim).to(device)
    else:
        model = ConditionalFlowMLP(A_dim, Z_dim).to(device)

    raw_state = torch.load(ckpt_path, map_location=device)
    clean_state = {k.replace("_orig_mod.", ""): v for k,v in raw_state.items()}
    model.load_state_dict(clean_state)
    model.eval()
    return model
def compute_state_error(ood_model, state, action, target, device):
    """
    Single-step error:
     - x1  = action_t
     - z   = [state_t, target]
     - x0  = Gaussian noise of same shape
     - t   = random diffusion time
     Returns mean abs error between model(x1,t,z) and x1.
    """
    A_dim = action.size
    Z_dim = state.size + target.size
    # make tensors
    x1 = torch.from_numpy(action).float().to(device).unsqueeze(0)    # [1,A]
    x0 = torch.randn_like(x1)                                        # [1,A]
    z  = torch.from_numpy(np.concatenate([state, target])).float()   # (Z,)
    z  = z.to(device).unsqueeze(0)                                   # [1,Z]
    t  = torch.rand(1, device=device)                                # [1]
    with torch.no_grad():
        v_pred = ood_model(x1, t, z).squeeze(0)                      # [A]
    err = (v_pred - x1.squeeze(0)).abs().mean().item()
    return err


def compute_log_likelihood(
    ood_model,
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    step_size: float = 0.1,
    method: str = 'midpoint'
) -> float:
    """
    Computes the average log-likelihood of a window of (state, action, target) pairs
    under a pre-trained multi-step Conditional CNF.

    Args:
      ood_model: a ConditionalFlowAttention or ConditionalFlowMLP instance
      history:   a list of length W = seq_len, each element is (state, action, target)
                 - state:  np.ndarray of shape (S,)
                 - action: np.ndarray of shape (A,)
                 - target: np.ndarray of shape (G,)
      device:    torch.device on which to run the model and solver
      step_size: step size for the ODE solver
      method:    integration method for the solver (e.g. 'midpoint', 'rk4')

    Returns:
      A single float: the mean log-likelihood of that window under the CNF.
    """
    # Number of timesteps in the window (should match the seq_len used in training)
    W = len(history)
    assert W > 0, "History must contain at least one timestep"

    # Dimensions
    S = history[0][0].shape[0]    # state dimension
    A = history[0][1].shape[0]    # action dimension
    G = history[0][2].shape[0]    # goal dimension

    # 1) Build the flattened action vector x1 of shape [1, W*A]
    a_seq = np.stack([a for (_, a, _) in history], axis=0)      # (W, A)
    x1 = torch.from_numpy(a_seq.reshape(1, W * A)).float().to(device)  # (1, W*A)

    # 2) Build the flattened context vector z of shape [1, W*(S+G)]
    s_seq = np.stack([s for (s, _, _) in history], axis=0)      # (W, S)
    g_seq = np.stack([g for (_, _, g) in history], axis=0)      # (W, G)
    z_seq = np.concatenate([s_seq, g_seq], axis=1)              # (W, S+G)
    z = torch.from_numpy(z_seq.reshape(1, W * (S + G))).float().to(device)  # (1, W*(S+G))

    # 3) Set up the base Gaussian prior over the flattened action space
    D = W * A
    base_dist = Independent(
        Normal(torch.zeros(D, device=device),
               torch.ones(D,  device=device)),
        1
    ).log_prob

    # 4) Wrap the OOD model into a velocity model for the solver
    wrapper = ConditionalVelocityWrapper(ood_model, z)  # z has shape [1, W*(S+G)]
    solver  = ODESolver(velocity_model=wrapper)

    # 5) Compute the likelihood via CNF ODE integration
    with torch.no_grad():
        _, logp = solver.compute_likelihood(
            x_1               = x1,          # shape (1, W*A)
            method            = method,
            step_size         = step_size,
            exact_divergence  = True,
            log_p0            = base_dist
        )
    # logp is a tensor of shape [1], return its item
    return logp.item()

#######################
# Replanning determinator
#######################
# 1) Hyperparameters for adaptive replanning
ts             = []
metrics_window = []
window_size    = 10
# thresholds
err_thresh     = 0.3
ll_thresh      = 0

start = time.time()
end   = None
L_vals = []
#######################
# Argument parsing
#######################
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config:  str = 'config.maze2d'

args = Parser().parse_args('plan')
Ns   = args.horizon
K = 50
#######################
# Environment & Policy setup
#######################
env = datasets.load_environment(args.dataset)
diff_exp = load_diffusion_manual(
    args.logbase, args.dataset, horizon=256, n_steps=256,
    epoch=args.diffusion_epoch, device=args.device
)
diffusion = diff_exp.ema
renderer  = diff_exp.renderer
policy    = Policy(diffusion, diff_exp.dataset.normalizer)
print(f"Evaluating with horizon=256, n_steps=256")

# instantiate your OOD model (attention or mlp)
A_dim, Z_dim = 20,60
ckpt_path    = "./checkpoints/cnf_multistep/cross_attention_checkpoints/epoch2000.pth"
ood_model    = init_ood_detector('attention', A_dim, Z_dim, ckpt_path, args.device)

#######################
# Main control loop
#######################
observation = env.reset(seed=seed)
state       = env.state_vector().copy()
if args.conditional:
    env.set_target()
target      = env._target

rollout     = [observation.copy()]
total_reward= 0.0
sequence    = None
plan_ptr    = 0

for t in range(1600):
    state = env.state_vector().copy()

    # 1) initialize plan
    if t == 0:
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
        _, samples = policy(cond,
                            batch_size=args.batch_size,
                            diffusion_steps=Ns,
                            replan_mode='scratch')
        sequence = samples.observations[0]
        plan_ptr = 0
        renderer.composite(
            join(args.savepath, f'plan_{t}.png'),
            samples.observations, ncol=1
        )
    if t == 100:
            print("Interfer starts")
            offset = maze2d.teleport_agent(env, level='medium')
            print(f"[t={t}] Teleported by {offset}")
            continue

    # 2) select next waypoint & compute action
    wp        = sequence[plan_ptr]
    action    = wp[:2] - state[:2] + (wp[2:] - state[2:])
    next_obs, reward, terminal, _ = env.step(action)
    total_reward += reward
    rollout.append(next_obs.copy())

    # 3) update window and compute metric
    metrics_window.append((state, action, np.array(target) ))
    if len(metrics_window) > window_size:
        metrics_window.pop(0)

    # choose ONE of these two to fill your windowed metric:
    # --- option A: state-action error per step ---
    # current_err = compute_state_error(ood_model, state, action, target, args.device)
    # metric = current_err

    # --- option B: CNF log-likelihood over window ---


    ts.append(t)

    # 4) check thresholds and replan if needed every K steps
    if (t % K == 0) and len(metrics_window) == window_size:
        current_ll  = compute_log_likelihood(ood_model, metrics_window, args.device)
        metric      = current_ll
        print("current likelihood",metric)
        if metric > err_thresh if 'err' in locals() else metric < ll_thresh:
            print(f"[t={t}] Replanning triggered (metric={metric:.3f})")
            cond = {
                0:                    state.copy(),
                diffusion.horizon-1: np.array([*target, 0, 0])
            }
            _, samples = policy(cond,
                                batch_size=args.batch_size,
                                diffusion_steps=Ns,
                                replan_mode='scratch')
            sequence = samples.observations[0]
            plan_ptr = 0
            renderer.composite(
                join(args.savepath, f'plan_replan_{t}.png'),
                samples.observations, ncol=1
            )
        L_vals.append(metric)

    # 5) advance pointer
    plan_ptr = min(plan_ptr + 1, len(sequence)-1)

    # 6) optional visualization every K steps
    if t % K == 0:
        renderer.composite(
            join(args.savepath, f'cur_wpt{t}.png'),
            np.expand_dims(sequence,0), ncol=1
        )
        renderer.composite(
            join(args.savepath, f'cur_rollout{t}.png'),
            np.array([rollout]), ncol=1
        )

    # 7) termination check
    if maze2d.check_done(env):
        print(f"🏁 Done at t={t}, return={total_reward:.2f}")

# final dumps
renderer.composite(
    join(args.savepath, 'final_rollout.png'),
    np.array([rollout]), ncol=1
)
with open(join(args.savepath, 'rollout.json'),'w') as f:
    json.dump({
        'step': t,
        'return': total_reward,
        'term': terminal,
        'score': env.get_normalized_score(total_reward)
    }, f, indent=2)

if end is None:
    end = time.time()
score = env.get_normalized_score(total_reward)
print(f"Elapsed: {end - start:.2f}s, Steps: {len(rollout)-1}, Return: {total_reward:.2f},,score = {score:.2f}")