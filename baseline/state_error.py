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

#######################
# Replanning determinator
#######################
# 1) Hyperparameters for adaptive replanning
ts = []       # list of timesteps
L_vals = []   # corresponding log‐likelihood values
horizon = 256   #trajectory length
n_steps = 256   #diffusion steps


start = time.time()
end = None
#######################
# Argument parsing
#######################
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config:  str = 'config.maze2d'

args = Parser().parse_args('plan')
Ns = args.horizon

#######################
# Environment & Policy setup
#######################
env = datasets.load_environment(args.dataset)

# Custom load of diffusion (to match your horizons)
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
observation  = env.reset(seed = seed) #42
state        = env.state_vector().copy()
if args.conditional:
    env.set_target()
target       = env._target

# PD controller parameters 
K            = 100     # replan every K steps
Kp           = 0.6    # P–controller gain
Kd           = 0.6            # tune this
prev_error = np.zeros(2)  
rollout      = [observation.copy()]
total_reward = 0.0
sequence     = None
plan_ptr     = 0
err_thresh = 0.3 # state error threshold
global_history = rollout.copy()
for t in range(1600): #env.max_episode_steps
    state = env.state_vector().copy()

    # 1) init plan
    if t == 0:
        print(f"[t={t}] Init from start {state[:2]} to target {target}")

        # build conditioning dict
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
        #breakpoint()
        _, samples = policy(cond, batch_size=args.batch_size,diffusion_steps = Ns,replan_mode = 'scratch')
        sequence   = samples.observations[0]   # (horizon, state_dim)
        plan_ptr   = 0
        # also save your composite if desired
        # renderer.composite(
        #     join(args.savepath, f'plan_{t}.png'),
        #     samples.observations,
        #     ncol=1
        # )
    wp_current = sequence[plan_ptr]
    pos_err    = np.linalg.norm(wp_current[:2] - state[:2])
    if pos_err > err_thresh and t% K ==0:
            print(f"[t={t}] Replanning triggered (error={pos_err:.3f} > {err_thresh})")
            # rebuild conditioning dict
            cond = {
                0:                            state.copy(),
                diff_exp.diffusion.horizon-1: np.array([*target, 0, 0])
            }
            _, samples = policy(
                cond,
                batch_size=args.batch_size,
                diffusion_steps=n_steps,
                replan_mode='scratch'
            )
            sequence = samples.observations[0]
            plan_ptr = 0
            # visualize the new plan
            # renderer.composite(join(args.savepath, f'plan_replan_{t}.png'),
            #                    samples.observations, ncol=1)
    # 2) Read current waypoint

    next_waypoint  = sequence[plan_ptr]        # [x,y,vx,vy]

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    if t == 100:
        print("Interfer starts")
        offset = maze2d.teleport_agent(env, level='medium')
        print(f"[t={t}] Teleported by {offset}")
        continue
    next_obs, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    rollout.append(next_obs.copy())
    global_history.append(next_obs.copy())
    # 4) Advance the pointer AFTER stepping
    plan_ptr = min(plan_ptr + 1, len(sequence)-1)
    ts.append(t)
    # for debuging (output current planning trajectory vs current rollouts)
    # if t% K ==0:
    #     current_waypoint = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
    #     renderer.composite(
    #             join(args.savepath, f'cur_wpt{t}.png'),
    #             current_waypoint,
    #             ncol=1
    #         )
    #     renderer.composite(
    #             join(args.savepath, f'cur_rollout{t}.png'),
    #              np.array([rollout]),
    #             ncol=1
    #         )
     # 5) Check for termination # FIXME not actually works
    if maze2d.check_done(env):
        print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
        end = time.time()
        #break
# 6) Final dump
# renderer.composite(
#     join(args.savepath, 'final_rollout.png'),
#     np.array([rollout]),
#     ncol=1
# )
with open(join(args.savepath, 'rollout.json'), 'w') as f:
    json.dump({
        'step':  t,
        'return': total_reward,
        'term':   terminal,
        'score':  env.get_normalized_score(total_reward)
    }, f, indent=2)
if end == None:
    end = time.time()
time_taken = start - end
print(f"Elapsed: {time_taken:.4f} s")
print(f"Done. Steps={len(rollout)-1}, Return={total_reward:.2f},Score = {score:.2f}")

