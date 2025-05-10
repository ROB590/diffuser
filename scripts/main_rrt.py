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
import math, random
np.set_printoptions(threshold=np.inf, linewidth=np.inf) # For debug
# Hyperparameter

########################### Replanning determinator
# 1) Hyperparameters for adaptive replanning
ls       = 1.2       # full‐replan threshold (tune on validation)
lf       = 0.7          # partial‐replan threshold (ls < lf)
I        = [50,100,200] # diffusion steps to sample for KL estimate #NOTE must smaller than the diffusion noise step #should be more than 1
# 2) Control paraemters
K            = 100     # replan every K steps
Kp           = 0.58     # P–controller gain 0.6
Kd = 0.7            # tune this  0.6
Ki = 0.001
dt = 0.05


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
def compute_L_t(diffusion, tau0, cond, I):
    """Average KL over timesteps in I for detection."""
    device = next(diffusion.parameters()).device
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
        kl = 0.5*((logvar_pred - logvar_true)
                + (torch.exp(logvar_true - logvar_pred))
                + (μ_true - μ_pred).pow(2)*torch.exp(-logvar_pred)
                - 1).mean()
        kl_vals.append(kl.item())
    L_t = sum(kl_vals) / len(kl_vals)
    return L_t
# RRT plan with diffusion
#NOTE default would be rrt connect if can not find the efficient path
def rrt(diffusion,policy,grid, start, goal):
    # calls the rrt_connect above
    print("real rrt start point is",start)
    path = rrt_connect(diffusion,policy,grid, start, goal,
                       max_iter=1000,
                       extend_len=0.5)
    if path is None:
        return None
    # convert cell-based points → continuous already done
    return np.array(path, dtype=np.float32)

def rrt_connect(diffusion, policy, grid, start, goal,
                max_iter=500, extend_len=1.0, horizon=384,
                I=[50,100,200], kl_thresh=0.8):
    class Node:
        __slots__ = ('x','y','parent')
        def __init__(self, x, y, parent=None):
            self.x, self.y, self.parent = x, y, parent

    def steering(from_node, to_point):
        dx, dy = to_point[0] - from_node.x, to_point[1] - from_node.y
        d = math.hypot(dx, dy)
        if d <= extend_len:
            return Node(to_point[0], to_point[1], from_node)
        theta = math.atan2(dy, dx)
        return Node(
            from_node.x + extend_len * math.cos(theta),
            from_node.y + extend_len * math.sin(theta),
            from_node
        )

    def kl_safe(a, b):
        """Try a diffusion rollout between a→b, return (is_safe, micro_plan)."""
        cond = {
            0:                   np.array([a.x,   a.y,   0, 0], dtype=np.float32),
            diffusion.horizon-1: np.array([b.x,   b.y,   0, 0], dtype=np.float32)
        }
        _, samples = policy(cond, batch_size=1)
        micro = samples.observations[0]  # [horizon, obs_dim]
        L = compute_L_t(diffusion, micro, cond, I)
        print(f'Check the sequence generate from {a.x,a.y} to {b.x,b.y},loglikelihood is {L}')
        micro_plan = np.expand_dims(micro, axis=0)
        renderer.composite(
                join(args.savepath, f'checking_kl.png'),
                micro_plan,
                ncol=1
            )
        #breakpoint()
        return (L > kl_thresh), micro[:, :2]  # return positions only

    def nearest(tree, pt):
        return min(tree, key=lambda n: (n.x - pt[0])**2 + (n.y - pt[1])**2)

    def build_path(node):
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    tree_s = [Node(*start)]
    tree_g = [Node(*goal)]

    for iter_idx in range(max_iter):
        # 1) Sample random point
        rnd = (random.uniform(0, grid.shape[1]),
               random.uniform(0, grid.shape[0]))

        # 2) Extend start‐tree
        near_s = nearest(tree_s, rnd)
        new_s  = steering(near_s, rnd)
        if detect_collisions_grid(env, [(new_s.x, new_s.y)])[0]:
            continue
        tree_s.append(new_s)

        # 3) Try connect goal‐tree toward new_s
        near_g = nearest(tree_g, (new_s.x, new_s.y))
        new_g  = steering(near_g, (new_s.x, new_s.y))
        if detect_collisions_grid(env, [(new_g.x, new_g.y)])[0]:
            continue
        tree_g.append(new_g)
        # ** diffusion‐KL check in lieu of collision‐only **
        safe, micro_path = kl_safe(new_s, new_g)
        if safe:
            # stitch: start‐tree path + micro‐plan + goal‐tree path
            path_from_start = build_path(new_s)
            path_from_goal  = build_path(new_g)
            #print("Path from start,",path_from_start)
            #print("Path from goal,",path_from_goal)
            #print("intermediate path",micro_path)
            # remove duplicate midpoint, then concatenate
            return (
                path_from_start
                + [(x, y) for (x, y) in micro_path]
                + path_from_goal[::-1][1:]
            )
        # otherwise, fall back: if direct connect (no collision), accept it
        if not detect_collisions_grid(env, [(new_g.x, new_g.y)])[0]:
            tree_g.append(new_g)
            if math.hypot(new_s.x - new_g.x, new_s.y - new_g.y) < 1e-6:
                # pure-RRT-Connect success
                #print("Default rrt success")
                #print("part 1 path:",build_path(new_s))
                #print("part 2 path",build_path(new_g)[::-1][1:])
                return build_path(new_s) + build_path(new_g)[::-1][1:]

        # 4) Swap trees
        tree_s, tree_g = tree_g, tree_s
        
    # no path found
    return None
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
        #print("Current horizon",horizon)
        #print('current step',n_steps)
    trainer.load(epoch)

    return DiffusionExperiment(
        dataset_obj, renderer, model, diffusion, trainer.ema_model, trainer, epoch
    )

# 2) Decision function  #FIXME rrt planner activate also should measure for ood?
def should_replan(diffusion, old_seq,rollout,current_planner, cond, t, ls, lf, I):
    if current_planner == 'diffusion':
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
    else:
        return "rrt_eval"
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
seed = 2
print(f"Evaluating with horizon={horizon}, n_steps={n_steps}")

#######################
# Main control loop
#######################
observation  = env.reset(seed=seed)
state        = env.state_vector().copy()
if args.conditional:
    env.set_target()
target       = env._target

prev_error = np.zeros(2) 
integral = np.zeros(2) 
rollout      = [observation.copy()]
total_reward = 0.0
sequence     = None
plan_ptr     = 0

current_planner = 'diffusion'
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
        # FIXME distance between rollout and sequence left over are too large  only replan based on current sequence if replan?
        mode = should_replan(diffusion, sequence,rollout,current_planner, cond, t, ls, lf, I)
    else:
        mode = None
    if mode == 'scratch':
        current_planner = 'rrt'
        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by RRT Connect")
        # 1) read start/goal in continuous space
        grid = env.unwrapped.maze_arr
        start = env.state_vector()[:2]
        goal  = cond[diffusion.horizon-1][:2]

        raw_xy = rrt(diffusion,policy,grid, start, goal)
        if raw_xy is None:
            raise RuntimeError("RRT failed to find a path")
        # 4) interpolate/pad/truncate to horizon
        plan_xy = fit_to_horizon(raw_xy, diffusion.horizon)

        obs_dim = dataset.observation_dim
        seq4 = np.zeros((diffusion.horizon, obs_dim), dtype=np.float32)
        seq4[:, :2] = plan_xy
        sequence = seq4
        plan_ptr = 0

        # visualize the RRT plan:
        rrtplan = np.expand_dims(sequence, axis=0)
        renderer.composite(
            join(args.savepath, f'plan_rrt_t{t}.png'),
            rrtplan,
            ncol=1
        )
    elif mode == 'future':
        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by future")
        # full replanning (Algorithm 2)
        state = env.state_vector().copy()
        cond[0] = state
        _, samples = policy(cond, batch_size=args.batch_size)
        sequence = samples.observations[0]
        fplan = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
        renderer.composite(
            join(args.savepath, f'plan_future_t{t}.png'),
            fplan,
            ncol=1
        )
        plan_ptr = 0
    elif mode == 'rrt_eval':
        print(f"[t={t}] Evaluating from start {state[:2]} to target {target} by for future diffuser trajectory")
        state = env.state_vector().copy()
        cond[0] = state
        # tau0 = sequence.copy()
        _, samples = policy(cond, batch_size=args.batch_size)
        tau0 = samples.observations[0]
        L_astar = compute_L_t(diffusion, tau0, cond, I)
        print(f"[t={t}] RRT*‐plan OOD‐score L_t = {L_astar:.3f}")
        if L_astar > ls:
            # switch back to diffusion
            print("  → switching back to diffusion planner")
            current_planner = 'diffusion'
            _, samples = policy(cond, batch_size=args.batch_size)
            sequence   = samples.observations[0]
            plan_ptr   = 0
            fplan = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
            renderer.composite(
                join(args.savepath, f'eval_back_td{t}.png'),
                fplan,
                ncol=1
            )
    
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