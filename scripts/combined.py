import os
import pickle
import json
import argparse

import numpy as np
from os.path import join
from heapq import heappush, heappop

import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.guides.policies import Policy
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch

# ----------------------------
# Grid-based collision detection
# ----------------------------
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

# ----------------------------
# Manual diffusion loader
# ----------------------------
def load_diffusion_manual(logbase, dataset_name, horizon, n_steps,
                          epoch='latest', device='cuda'):
    base = os.path.join(logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')
    cfgs = {}
    for name in ['dataset','render','model','diffusion','trainer']:
        with open(join(base, f'{name}_config.pkl'), 'rb') as f:
            cfgs[name] = pickle.load(f)
    train_ds = cfgs['dataset']()
    model = cfgs['model']().to(device)
    diffusion = cfgs['diffusion'](model).to(device)
    trainer = cfgs['trainer'](diffusion, train_ds, cfgs['render']())
    if epoch == 'latest':
        epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}'))
    trainer.load(epoch)
    return DiffusionExperiment(
        train_ds, cfgs['render'](), model, diffusion,
        trainer.ema_model, trainer, epoch
    )

# ----------------------------
# Main execution
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',           type=str, default='maze2d-large-v1')
    parser.add_argument('--eval_dataset',      type=str, default='maze2d-medium-v1')
    parser.add_argument('--horizon',           type=int, default=384)
    parser.add_argument('--n_diffusion_steps', type=int, default=256)
    parser.add_argument('--diffusion_epoch',   type=str, default='latest')
    parser.add_argument('--logbase',           type=str, default='logs')
    parser.add_argument('--savepath',          type=str, default='results')
    parser.add_argument('--vis_freq',          type=int, default=50)
    parser.add_argument('--device',            type=str, default='cuda')
    parser.add_argument('--batch_size',        type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.savepath, exist_ok=True)

    # 1) load environment & renderer
    eval_env = datasets.load_environment(args.eval_dataset)
    renderer = utils.Maze2dRenderer(args.eval_dataset)

    # 2) load diffusion policy
    diff_exp = load_diffusion_manual(
        args.logbase,
        args.dataset,
        args.horizon,
        args.n_diffusion_steps,
        args.diffusion_epoch,
        args.device
    )
    policy = Policy(diff_exp.ema, diff_exp.dataset.normalizer)

    # 3) sample the raw diffusion trajectory
    obs0 = eval_env.reset(seed=42)
    if hasattr(eval_env, 'set_target'):
        eval_env.set_target()
    target = eval_env.unwrapped._target
    cond = {args.horizon-1: np.array([*target,0,0]), 0: obs0}
    _, samples = policy(cond, batch_size=args.batch_size)
    seq = samples.observations[0]
    pos_seq = seq[:, :2]

    # save the original diffusion plan
    renderer.composite(
        join(args.savepath, 'original_plan.png'),
        samples.observations,
        ncol=1
    )
    print(f"Saved raw diffusion plan → {args.savepath}/original_plan.png")
    print(f"Original trajectory length: {len(pos_seq)}")
    print(f"Original trajectory: {pos_seq}")
    # 4) detect collisions and hybrid-splice with A*
    collisions = detect_collisions_grid(eval_env, pos_seq)
    cell_seq = [to_cell(p) for p in pos_seq]
    grid = eval_env.unwrapped.maze_arr

    final_seq = []
    i, H = 0, len(pos_seq)
    while i < H-1:
        final_seq.append(np.array([*pos_seq[i],0.0,0.0]))
        if collisions[i+1]:
            # find next non-colliding index
            j = next(k for k in range(i+1, H) if not collisions[k])
            print(f"Collision detected at {i+1}, splicing A* from {i} to {j}")
            print('Current pos_sequence befor point', pos_seq[i])
            print('Current pos_sequence after point', pos_seq[j])
            sub_cells = astar(grid, cell_seq[i], cell_seq[j])
            print("before cell_seq", cell_seq[i])
            print("after cell_seq", cell_seq[j])
            cont = [np.array([*cell_to_world(c),0.0,0.0]) for c in sub_cells]
            final_seq.extend(cont[1:-1])
            i = j
        else:
            i += 1
    final_seq.append(np.array([*pos_seq[-1],0.0,0.0]))
    print(f"Final trajectory length: {len(final_seq)}")
    print(f"Final trajectory: {final_seq}")
    # save the A*-fixed plan
    fixed_obs = np.stack(final_seq)[None]  # shape (1, L, 4)
    renderer.composite(
        join(args.savepath, 'fixed_plan.png'),
        fixed_obs,
        ncol=1
    )
    print(f"Saved A*-fixed plan → {args.savepath}/fixed_plan.png")

    # 5) execute the final trajectory
    tol = 0.05                  # threshold for “close enough” (tune as needed)
    max_steps = eval_env.max_episode_steps

    # initialize
    obs    = obs0
    rollout = [obs0.copy()]
    actions = []
    total_reward = 0.0
    wp_idx = 0
    tol = 0.08
    for t in range(800):

        state = eval_env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = obs

            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
        #pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            #pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        #pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = eval_env.step(action)
        total_reward += reward
        score = eval_env.get_normalized_score(total_reward)
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'{action}'
        )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = eval_env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f'{t}.png')

            if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


            #renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # 6) save rollout statistics
    with open(join(args.savepath, 'stats.json'), 'w') as f:
        json.dump({'return': total_reward, 'steps': len(rollout)-1}, f, indent=2)
    print(f"Done! return={total_reward:.2f}, steps={len(rollout)-1}")

if __name__ == '__main__':
    main()
