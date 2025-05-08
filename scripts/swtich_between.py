import os
import json
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.models import TemporalUnet, GaussianDiffusion
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch
import torch
np.set_printoptions(threshold=np.inf, linewidth=np.inf) # For debug
# convert between astar 
def fit_to_horizon(path_xy, horizon):
    """
    path_xy: np.array of shape (L,2), arbitrary L
    returns: np.array of shape (horizon,2)
    """
    L = path_xy.shape[0]
    if L == horizon:
        return path_xy.copy()
    # parametrize original path by u in [0,1]
    u_orig = np.linspace(0.0, 1.0, L)
    u_new  = np.linspace(0.0, 1.0, horizon)
    # interpolate x and y separately
    x_new = np.interp(u_new, u_orig, path_xy[:,0])
    y_new = np.interp(u_new, u_orig, path_xy[:,1])
    return np.stack([x_new, y_new], axis=1)
# Astar helper ----------
WALL = 10

def to_cell(pt):
    """Convert continuous (x,y) to integer grid cell."""
    return int(round(pt[0])), int(round(pt[1]))

def cell_to_world(cell):
    """Convert grid cell (i,j) back to continuous (x,y) at cell center."""
    i, j = cell
    return np.array([i,j])

def detect_collisions_grid(env, positions):
    """
    Detect collisions by checking occupancy grid cells.
    Returns list of bools for each (x,y) in positions.
    """
    grid = env.unwrapped.maze_arr
    flags = []
    H, W = grid.shape
    for pos in positions:
        cell = to_cell(pos)
        if not (0 <= cell[0] < H and 0 <= cell[1] < W) or grid[cell] == WALL:
            flags.append(True)
        else:
            flags.append(False)
    return flags

# ----------------------------
# A* on occupancy grid
# ----------------------------
def astar(grid: np.ndarray, start: tuple, goal: tuple):
    """4-connected grid A* from start to goal"""
    H, W = grid.shape
    assert grid[start] != WALL and grid[goal] != WALL
    neigh = [(1,0),(-1,0),(0,1),(0,-1)]
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start,goal), 0, start)]
    came = {}
    gscore = {start:0}
    closed = set()
    while open_set:
        _, g, cur = heappop(open_set)
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        closed.add(cur)
        for dx, dy in neigh:
            nb = (cur[0]+dx, cur[1]+dy)
            if not (0<=nb[0]<H and 0<=nb[1]<W):
                continue
            if grid[nb] == WALL or nb in closed:
                continue
            tg = g + 1
            if tg < gscore.get(nb, float('inf')):
                came[nb] = cur
                gscore[nb] = tg
                heappush(open_set, (tg + h(nb, goal), tg, nb))
    return None

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
ls       = 1       # full‐replan threshold (tune on validation)
lf       = 0.9          # partial‐replan threshold (ls < lf)
I        = [50,100,200] # diffusion steps to sample for KL estimate #NOTE must smaller than the diffusion noise step #should be more than 1

# 2) Decision function
def should_replan(diffusion, old_seq,rollout, cond, t, ls, lf, I):
    device = next(diffusion.parameters()).device
    # 2.1 Build partial trajectory tau0 with real observations
    tau0 = old_seq.copy()
    for k in range(1, t+1):
        tau0[k] = rollout[k] # replace with rollouts
    assert tau0.shape[0]==horizon
    test = np.expand_dims(tau0, axis=0)   # sequence is your H×obs_dim A* plan
    renderer.composite(
            join(args.savepath, f'measure_ood_t{t}.png'),
            test,
            ncol=1
        )
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
observation  = env.reset()
state        = env.state_vector().copy()
if args.conditional:
    env.set_target()
target       = env._target

K            = 50     # replan every K steps
Kp           = 0.58     # P–controller gain 0.6
Kd = 0.7            # tune this  0.6
Ki = 0.001
prev_error = np.zeros(2) 
integral = np.zeros(2) 
dt = 0.05
rollout      = [observation.copy()]
total_reward = 0.0
sequence     = None
plan_ptr     = 0

for t in range(800): #env.max_episode_steps
    state = env.state_vector().copy()

    # 1) init plan
    if t == 0:
        print(f"[t={t}] Init from start {state[:2]} to target {target}")

        # build conditioning dict
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
        _, samples = policy(cond, batch_size=args.batch_size)
        sequence   = samples.observations[0]   # (horizon, state_dim)
        #sequence = sequence[1:]  #now shape (H-1, obs_dim)
        plan_ptr   = 1

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
        #print('current seq',sequence)
    if t > 0 and t < diffusion.horizon and (t % K) == 0:
        mode = should_replan(diffusion, sequence,rollout, cond, t, ls, lf, I)
    else:
        mode = None
    # if mode == 'scratch':
    #     print(f"[t={t}] Replanning from start {state[:2]} to target {target} by scratch")
    #     # full replanning (Algorithm 2)
    #     _, samples = policy(cond, batch_size=args.batch_size)
    #     sequence = samples.observations[0]
    #     plan_ptr = 0
    if mode == 'scratch':
        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by A*")
        # 1) read start/goal in continuous space
        start = env.state_vector()[:2]                       # e.g. [x,y,…]
        goal  = cond[diffusion.horizon - 1][:2]

        # 2) grid-based A*
        grid = env.unwrapped.maze_arr
        start_cell = to_cell(start)
        goal_cell  = to_cell(goal)
        cell_path = astar(grid, start_cell, goal_cell)

        if cell_path is not None:
            # 3) convert cells → world (x,y)
            raw_xy = np.array([cell_to_world(c) for c in cell_path], dtype=np.float32)
            # 4) interpolate/compress to exactly `horizon` points
            plan_xy = fit_to_horizon(raw_xy, diffusion.horizon)

            # 5) build full state‐trajectory [H × obs_dim]
            obs_dim = dataset.observation_dim
            seq4 = np.zeros((diffusion.horizon, obs_dim), dtype=np.float32)
            seq4[:, :2] = plan_xy
            # leave other dimensions (e.g. velocity) at zero
            sequence = seq4
        else:
            # fallback to diffusion if A* fails
            print("[WARNING] A* found no path → using diffusion fallback")
            _, samples = policy(cond, batch_size=args.batch_size)
            sequence = samples.observations[0]
        # visualize
        aplan = np.expand_dims(sequence, axis=0)   # sequence is your H×obs_dim A* plan
        renderer.composite(
            join(args.savepath, f'plan_astar_t{t}.png'),
            aplan,
            ncol=1
        )
        plan_ptr = 0
    elif mode == 'future':
        # print(f"[t={t}] Replanning from start {state[:2]} to target {target} by future")
        # # partial replanning (Algorithm 3):
        # # Keep states up to t, regenerate future tail
        # cond_new = {0: sequence[t], diffusion.horizon-1: cond[diffusion.horizon-1]}
        # _, samples_fut = policy(cond_new, batch_size=args.batch_size)
        # # splice new future onto executed prefix
        # horizon = diffusion.horizon
        # new_tail = samples_fut.observations[0][t:]     
        # sequence = np.concatenate([
        #     sequence[:t],    
        #     new_tail], axis=0)         
        # plan_ptr = 0

        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by future")
        # full replanning (Algorithm 2)
        _, samples = policy(cond, batch_size=args.batch_size)
        sequence = samples.observations[0]
        fplan = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
        renderer.composite(
            join(args.savepath, f'plan_future_t{t}.png'),
            fplan,
            ncol=1
        )
        plan_ptr = 0
    
    if t% K ==0:
        current_waypoint = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
        renderer.composite(
                join(args.savepath, f'cur_wpt{t}.png'),
                current_waypoint,
                ncol=1
            )
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
        deriv = (error - prev_error) / 0.05    
    integral += error * dt  
    action = Kp * error + Kd * deriv + Ki*integral          
    prev_error = error.copy()
    # print(f"current at {env.state_vector()}", f"way point aim at{wp}", f"give an action{action}")
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
plan_batch = np.expand_dims(sequence, axis=0)  
#print("final rollout is", rollout)
# save a bird’s‐eye of the *plan* (blue line = planned path)
plan_path = join(args.savepath, 'final_plan.png')
renderer.composite(
    plan_path,
    plan_batch,
    ncol=1
)
print('final path is',plan_path)