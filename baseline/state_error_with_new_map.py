#!/usr/bin/env python
import os
import json
import pickle
import time
from os.path import join

import numpy as np
import torch
import gym
import imageio

import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.models import TemporalUnet, GaussianDiffusion
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch

# your helper functions / wrappers
from environment import maze2d
from environment.maze2d import (
    load_custom_env,
    Maze2dRenderer,
    within_bounds,
    in_collision,
    teleport_agent,
)

#######################
# Helper: load diffusion
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
        epoch = 980000 #200000
        print("Current horizon",horizon)
        print('current step',n_steps)
    trainer.load(epoch)

    return DiffusionExperiment(
        dataset_obj, renderer, model, diffusion, trainer.ema_model, trainer, epoch
    )

ts = []       # list of timesteps
L_vals = []   # corresponding log‐likelihood values
horizon = 256   #trajectory length
n_steps = 256   #diffusion steps
start = time.time()
end = None
#######################
# CLI & args
#######################
class Parser(utils.Parser):
    dataset:        str = 'maze2d-large-v1'
    config:         str = 'config.maze2d'
    logbase:        str = './logs'
    diffusion_epoch:str = 'latest'
    device:         str = 'cuda'
    batch_size:     int = 16
    savepath:       str = './results'
    custom_env:     str = 'maze-editv0-large'   # set to 'maze-editv1-large' for custom D4RL maze

args = Parser().parse_args('plan')
os.makedirs(args.savepath, exist_ok=True)
Ns = args.horizon
#######################
# Seed
#######################
seed = maze2d.ensure_seed(1)

#######################
# Environment setup
#######################
if args.custom_env:
    print("→ loading custom env:", args.custom_env)
    base_env = load_custom_env(args.custom_env, max_episode_steps=1600)
    # instantiate renderer early so we can extract its goal_position
    renderer = Maze2dRenderer(args.custom_env)
    # # grid coords: (row, col)
    # gy, gx = renderer.goal_position
    # # convert to continuous [x, y]
    # continuous_goal = np.array([gx + 0.5, gy + 0.5], dtype=np.float32)
    # # assign to env so env._target exists
    # base_env._target = continuous_goal
    env = base_env
    env._target = base_env._target
else:
    # raise RuntimeError('not correct loop')
    env = datasets.load_environment(args.dataset)
    # env = maze2d.ExternalDisturbanceWrapper(env=base_env, disturb_type="action")

#######################
# Diffusion & renderer setup
#######################
horizon = args.horizon
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

# If using the standard D4RL env, use its renderer; otherwise, we already built our custom renderer above
if not args.custom_env:
    renderer = diff_exp.renderer

policy = Policy(diffusion, dataset.normalizer)
print(f"→ Evaluating with horizon={horizon}, n_steps={n_steps}")

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
disturb_flag = False
for t in range(1600): #env.max_episode_steps
    state = env.state_vector().copy()
    if state[0]>3 and state[0]<4  and state[1]>3 and state[1]<4 and disturb_flag == False:
        print("disturbed!")
        disturb_flag = True
        sim_state = env.unwrapped.sim.get_state()
        sim_state.qpos[:2] = [4,3]
        env.unwrapped.sim.set_state(sim_state)
        env.unwrapped.sim.forward()
        continue
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
        renderer.composite(
            join(args.savepath, f'plan_{t}.png'),
            samples.observations,
            ncol=1
        )
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
            renderer.composite(join(args.savepath, f'plan_replan_{t}.png'),
                               samples.observations, ncol=1)
    # 2) Read current waypoint

    next_waypoint  = sequence[plan_ptr]        # [x,y,vx,vy]

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    # if t == 100:
    #     print("Interfer starts")
    #     offset = maze2d.teleport_agent(env, level='medium')
    #     print(f"[t={t}] Teleported by {offset}")
        # continue
    next_obs, reward, terminal, _ = env.step(action)
    total_reward += reward
    #score = env.get_normalized_score(total_reward)
    rollout.append(next_obs.copy())
    global_history.append(next_obs.copy())
    # 4) Advance the pointer AFTER stepping
    plan_ptr = min(plan_ptr + 1, len(sequence)-1)
    ts.append(t)
    # for debuging (output current planning trajectory vs current rollouts)
    if t% K ==0:
        current_waypoint = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
        renderer.composite(
                join(args.savepath, f'cur_wpt{t}.png'),
                current_waypoint,
                ncol=1
            )
        renderer.composite(
                join(args.savepath, f'cur_rollout{t}.png'),
                 np.array([rollout]),
                ncol=1
            )
     # 5) Check for termination # FIXME not actually works
    if maze2d.check_done(env):
        print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
        end = time.time()
        #break
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
        # 'score':  env.get_normalized_score(total_reward)
    }, f, indent=2)
if end == None:
    end = time.time()
time_taken = start - end
print(f"Elapsed: {time_taken:.4f} s")
print(f"Done. Steps={len(rollout)-1}, Return={total_reward:.2f}")

