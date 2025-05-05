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
        epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}'))
    trainer.load(epoch)

    return DiffusionExperiment(
        dataset_obj, renderer, model, diffusion, trainer.ema_model, trainer, epoch
    )


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
observation  = env.reset()
state        = env.state_vector().copy()
if args.conditional:
    env.set_target()
target       = env._target

K            = 50     # replan every K steps
Kp           = 1.0     # P–controller gain

rollout      = [observation.copy()]
total_reward = 0.0
sequence     = None
plan_ptr     = 0

for t in range(400): #env.max_episode_steps
    state = env.state_vector().copy()

    # 1) Replan every K steps
    if t % K == 0:
        print(f"[t={t}] Replanning from start {state[:2]} to target {target}")

        # build conditioning dict
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
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

    # 2) Read current waypoint
    wp          = sequence[plan_ptr]        # [x,y,vx,vy]
    pos_target  = wp[:2]
    pos_current = state[:2]

    # 3) Simple P–control on position
    action      = Kp * (pos_target - pos_current)
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