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

#######################
# Helper to load your diffusion experiment
#######################
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
        epoch = 200000
        print("Current horizon",horizon)
        print('current step',n_steps)
    trainer.load(epoch)

    return DiffusionExperiment(
        dataset_obj, renderer, model, diffusion, trainer.ema_model, trainer, epoch
    )

########################### Replanning determinator
# 1) Hyperparameters for adaptive replanning
ls       = 98       # full‐replan threshold (tune on validation)
lf       = 0          # partial‐replan threshold (ls < lf)
I        = [10, 50, 100] # diffusion steps to sample for KL estimate #NOTE must smaller than the diffusion noise step

# 2) Decision function
def should_replan(diffusion, old_seq, cond, t, ls, lf, I):
    device = next(diffusion.parameters()).device
    # 2.1 Build partial trajectory tau0 with real observations
    tau0 = old_seq.copy()
    for k in range(1, t+1):
        tau0[k] = env.state_vector()  # replace with actual observed state

    # 2.2 Estimate average KL over selected timesteps
    kl_vals = []
    for i in I:
        idx = torch.full((1,), i, dtype=torch.long, device=device)
        x0 = torch.tensor(tau0).float().unsqueeze(0).to(device)
        # forward noise
        noise = torch.randn_like(x0)
        x_i = diffusion.q_sample(x0, idx, noise=noise)
        # true posterior
        μ_true, _, logvar_true = diffusion.q_posterior(x0, x_i, idx)
        # model posterior
        μ_pred, var_pred, logvar_pred = diffusion.p_mean_variance(x_i, cond, idx)
        # KL divergence per Eq. (11)
        kl = 0.5*((logvar_pred - logvar_true)
                  + (torch.exp(logvar_true - logvar_pred))
                  + (μ_true - μ_pred).pow(2)*torch.exp(-logvar_pred)
                  - 1).mean()
        kl_vals.append(kl.item())
    L_t = sum(kl_vals) / len(kl_vals)

    # 2.3 Apply thresholds
    print(f"Average loglikelihood at {t}, is {L_t}")
    if L_t <= ls:
        return 'scratch'
    elif L_t <= lf:
        return 'future'
    else:
        return 'none'
#######################
# Argument parsing
#######################
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config:  str = 'config.maze2d'

args = Parser().parse_args('plan')


#######################
# Environment & Policy setup
#######################
env = datasets.load_environment(args.dataset)

# Custom load of diffusion (to match your horizons)
horizon = 384
n_steps = 256
diff_exp = load_diffusion_manual(
    args.logbase,
    args.dataset,
    horizon=horizon,
    n_steps=n_steps,
    epoch=args.diffusion_epoch,
    device=args.device
)
diffusion = diff_exp.ema
dataset   = diff_exp.dataset
renderer  = diff_exp.renderer
policy    = Policy(diffusion, dataset.normalizer)

print(f"Evaluating with horizon={horizon}, n_steps={n_steps}")

#######################
# Main control loop
#######################
observation  = env.reset(seed = 42)
state        = env.state_vector().copy()
if args.conditional:
    env.set_target()
target       = env._target

K            = 50     # replan every K steps
Kp           = 7     # P–controller gain
Kd = 0.8            # tune this
prev_error = np.zeros(2)  
rollout      = [observation.copy()]
total_reward = 0.0
sequence     = None
plan_ptr     = 0

for t in range(400): #env.max_episode_steps
    state = env.state_vector().copy()

    # 1) init plan
    if t == 0:
        print(f"[t={t}] Init from start {state[:2]} to target {target}")

        # build conditioning dict
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
        breakpoint()
        _, samples = policy(cond, batch_size=args.batch_size)
        sequence   = samples.observations[0]   # (horizon, state_dim)
        plan_ptr   = 0

        # plot the (x,y) path
        plan_xy = sequence[:, :2]              # extract positions
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(plan_xy[:,0], plan_xy[:,1], '-o', markersize=3, label='plan')
        ax.scatter(plan_xy[0,0], plan_xy[0,1], s=50, marker='*', label='start')
        ax.scatter(plan_xy[-1,0], plan_xy[-1,1], s=50, marker='X', label='target')
       
        ax.set_xlim(0,12)
        ax.set_ylim(0,12)

        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f"Replan @ t={t}")
        ax.legend()
        plot_path = join(args.savepath, f'replan_plot_{t}.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f" → saved replan plot to {plot_path}")

        # also save your composite if desired
        renderer.composite(
            join(args.savepath, f'plan_{t}.png'),
            samples.observations,
            ncol=1
        )
    if t > 0 and t < diffusion.horizon and (t % K) == 0:
        mode = should_replan(diffusion, sequence, cond, t, ls, lf, I)
    else:
        mode = None
    if mode == 'scratch':
        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by scratch")
        # full replanning (Algorithm 2)
        _, samples = policy(cond, batch_size=args.batch_size)
        sequence = samples.observations[0]
        plan_ptr = 0
    elif mode == 'future':
        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by future")
        # partial replanning (Algorithm 3):
        # Keep states up to t, regenerate future tail
        cond_new = {0: sequence[t], diffusion.horizon-1: cond[diffusion.horizon-1]}
        _, samples_fut = policy(cond_new, batch_size=args.batch_size)
        # splice new future onto executed prefix
        horizon = diffusion.horizon
        new_tail = samples_fut.observations[0][t:]     
        sequence = np.concatenate([
            sequence[:t],    
            new_tail], axis=0)         
        plan_ptr = 0
    # 2) Read current waypoint
    wp          = sequence[plan_ptr]        # [x,y,vx,vy]
    pos_target  = wp[:2]
    pos_current = state[:2]

    # 3) Simple P–control on position
    # action      = Kp * (pos_target - pos_current)
    error       = pos_target - pos_current       # [2]
    if t == 0:
        # no previous error yet → zero derivative
        deriv = np.zeros_like(error)
    else:
        deriv = (error - prev_error) / 1.0      
    action = Kp * error + Kd * deriv           
    prev_error = error.copy()
    next_obs, reward, terminal, _ = env.step(action)
    total_reward += reward
    rollout.append(next_obs.copy())

    # 4) Advance the pointer AFTER stepping
    plan_ptr = min(plan_ptr + 1, len(sequence)-1)

    # 5) Check for termination
    if terminal:
        print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
        break

# 6) Final dump
renderer.composite(
    join(args.savepath, 'final_rollout.png'),
    np.array([rollout]),
    ncol=1
)
with open(join(args.savepath, 'rollout.json'), 'w') as f:
    json.dump({
        'step':  t,
        'return': total_reward,
        'term':   terminal,
        'score':  env.get_normalized_score(total_reward)
    }, f, indent=2)

print(f"Done. Steps={len(rollout)-1}, Return={total_reward:.2f}")