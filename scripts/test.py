import argparse
import json
import os
from os.path import join

import math
import random
import numpy as np
from heapq import heappush, heappop

import diffuser.datasets as datasets
import diffuser.utils as utils   # Maze2dRenderer comes from here

# ----------------------------
# 1. A* 搜索（不变）
# ----------------------------
WALL  = 10
EMPTY = 11

def to_cell(pt):
    return int(round(pt[0])), int(round(pt[1]))
def cell_to_world(cell):
    # if each grid cell is 1×1 in world units and origin at (0,0):
    i, j = cell
    return np.array([j + 0.5, i + 0.5])
def astar(grid: np.ndarray, start: tuple, goal: tuple):
    """
    Find shortest path on a 4-connected grid from start to goal using A*.
    grid: 2D array of ints (10=wall, 11=free, 12=goal)
    start, goal: (i,j) integer cells
    Returns: list of (i,j) from start to goal, or None if no path.
    """
    H, W = grid.shape

    # sanity check
    assert 0 <= start[0] < H and 0 <= start[1] < W, "Start out of bounds"
    assert 0 <= goal[0]  < H and 0 <= goal[1]  < W, "Goal out of bounds"
    assert grid[start] != WALL, "Start lies inside a wall!"
    assert grid[goal]  != WALL, "Goal lies inside a wall!"

    # 4-connected neighbors
    neigh = [(1,0), (-1,0), (0,1), (0,-1)]
    # Manhattan heuristic :contentReference[oaicite:1]{index=1}
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = [(h(start, goal), 0, start)]
    came     = {}
    gscore   = {start: 0}
    closed   = set()

    while open_set:
        _, g, cur = heappop(open_set)
        if cur == goal:
            # reconstruct path
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            print("reach target")
            return path[::-1]

        closed.add(cur)
        for dx, dy in neigh:
            nb = (cur[0]+dx, cur[1]+dy)
            # skip out-of-bounds
            if not (0 <= nb[0] < H and 0 <= nb[1] < W):
                continue
            # **correct** obstacle check: skip if it's a WALL cell :contentReference[oaicite:2]{index=2}
            if grid[nb] == WALL or nb in closed:
                continue

            tg = g + 1
            if tg < gscore.get(nb, float('inf')):
                came[nb]    = cur
                gscore[nb]  = tg
                heappush(open_set, (tg + h(nb, goal), tg, nb))

    # no path found
    return None
# ----------------------------
# 2. 内置 RRT 实现
# ----------------------------
#FIXME rrt contain bugs 
def rrt(grid, start, goal, max_iter=2000, step_size=1.0, goal_sample_rate=0.05):
    """
    grid: 2D numpy array (1=obstacle, 0=free)
    start, goal: (i,j) integer grid coords
    Returns: list of (i,j) waypoints or None
    """
    nodes = [start]
    parent = {start: None}
    H, W = grid.shape

    def sample_free():
        # Goal bias: occasionally return goal directly
        if random.random() < goal_sample_rate:
            return goal
        while True:
            x = random.randint(0, H-1)
            y = random.randint(0, W-1)
            if grid[x, y] == 0:
                return (x, y)

    def nearest(pt):
        # Linear scan for nearest neighbor
        return min(nodes, key=lambda n: math.hypot(n[0]-pt[0], n[1]-pt[1]))

    def steer(frm, to):
        dx, dy = to[0]-frm[0], to[1]-frm[1]
        dist = math.hypot(dx, dy)
        if dist <= step_size:
            return to
        frac = step_size / dist
        return (int(round(frm[0] + dx*frac)),
                int(round(frm[1] + dy*frac)))

    def collision_free(p, q):
        # Bresenham’s line / interpolation collision check
        x0, y0 = p; x1, y1 = q
        steps = max(abs(x1-x0), abs(y1-y0))
        for t in range(steps+1):
            x = int(round(x0 + (x1-x0)*t/steps))
            y = int(round(y0 + (y1-y0)*t/steps))
            if grid[x, y] == 1:
                return False
        return True

    for _ in range(max_iter):
        q_rand = sample_free()
        q_near = nearest(q_rand)
        q_new  = steer(q_near, q_rand)
        if not collision_free(q_near, q_new):
            continue
        nodes.append(q_new)
        parent[q_new] = q_near

        # Check if we’ve reached the goal
        if math.hypot(q_new[0]-goal[0], q_new[1]-goal[1]) < step_size:
            path = [q_new]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
    return None

# ----------------------------
# 3. 主程序入口
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--planner', choices=['astar','rrt'], default='astar')
    p.add_argument('--dataset',  type=str, default='maze2d-large-v1')
    p.add_argument('--max-iter', type=int, default=2000)
    p.add_argument('--out-dir',  type=str, default='results')
    p.add_argument('--vis-freq', type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载环境 & 渲染器
    env = datasets.load_environment(args.dataset)
    renderer = utils.Maze2dRenderer(args.dataset)

    # 1) overwrite start & goal
    start_cell = np.array([5, 2], dtype=int)
    goal_cell  = np.array([7,10], dtype=int)


    # 1) reset to start cell
    # origin target and start
    obs0 = env.reset(seed = 42)
    grid  = env.unwrapped.maze_arr
    start = to_cell(obs0[:2])
    goal  = to_cell(env.unwrapped.get_target())
    print("origin target",goal)
    print("origin start",start)
    print("origin grid",grid)
    # obs0 = env.unwrapped.reset_to_location(start_cell)
    # env.unwrapped.set_target(goal_cell)
    # # 2) build grid & convert to planner indices
    # grid  = env.unwrapped.maze_arr       # the integer grid from parse_maze(…) :contentReference[oaicite:7]{index=7}
    # obs0 = env.unwrapped.reset_to_location(start_cell)
    # env.unwrapped.set_target(goal_cell)
    # obs0 = env.state_vector()
    # grid = env.unwrapped.maze_arr
    # start, goal = tuple(start_cell), tuple(goal_cell)
    # start = to_cell(obs0[:2])
    # goal  = to_cell(env.unwrapped.get_target())
    # print('new start', start)
    # print('new goal', goal)
    # print("new grid",grid)
    # 选择规划器
    if args.planner == 'astar':
        cell_path = astar(grid, start, goal)
    else:
        cell_path = rrt(
            grid, start, goal,
            max_iter=args.max_iter,
            step_size=1.0,
            goal_sample_rate=0.05
        )

    if cell_path is None:
        raise RuntimeError("路径规划失败")
    else:
        print('Find path',cell_path)
    # 离散轨迹 → 连续状态 [x,y,0,0]
    waypoints = [(float(i), float(j)) for i,j in cell_path]
    sequence  = [np.array([x, y, 0.0, 0.0]) for x,y in waypoints]
    # remove first one no need to go to start
    waypoints = np.array(waypoints)[1:]
    sequence = np.array(sequence)[1:]
    # 2) 执行并滚动可视化
    rollout = [obs0.copy()]
    actions = []
    total_r = 0.0
    wp_idx = 0
    tol = 0.05
    for t in range(env.max_episode_steps):
        pos = env.state_vector()[:2]
        if np.linalg.norm(pos - waypoints[wp_idx]) < tol and wp_idx < len(waypoints)-1:
            print("reach waypoint",wp_idx)
            wp_idx += 1
        wp  = sequence[wp_idx]
        st  = env.state_vector()
        pos, vel = st[:2], st[2:4]

        # PD gains (tune these!)
        kp = 7
        kd = 0.5

        # compute force = kp * position_error − kd * velocity
        force = kp * (wp[:2] - pos) - kd * vel

        a = np.clip(force, env.action_space.low, env.action_space.high)
        # teleport
        # qpos = env.unwrapped.sim.data.qpos.copy()    # full MuJoCo position vector
        # qvel = env.unwrapped.sim.data.qvel.copy()    # full MuJoCo velocity vector
        # qpos[0], qpos[1] = p[0], p[1]               # override x,y
        # qvel[:] = 0                                 # zero out velocities if desired
        # env.unwrapped.sim.set_state(qpos, qvel)
        # env.unwrapped.sim.forward()
        # obs = env.state_vector()  
        obs, reward, term, _ = env.step(a)
        rollout.append(obs.copy())
        actions.append(a.copy())
        total_r += reward

        if t % args.vis_freq == 0 or term:
            fullpath = join("results/", f'{t}.png')

            if t == 0: renderer.composite(fullpath, np.array([[obs]]), ncol=1)

            ## save rollout thus far
            renderer.composite(join("results/", 'rollout.png'), np.array(rollout)[None], ncol=1)

        if term:
            print('success terminate')
            break

    # 3) Summary
    score = env.get_normalized_score(total_r)
    result = {
        'planner': args.planner,
        'dataset': args.dataset,
        'return': float(total_r),
        'score': float(score),
        'steps': len(rollout)-1,
    }
    with open(join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"完成！ return={total_r:.2f}, score={score:.3f}, steps={len(rollout)-1}")

if __name__ == '__main__':
    main()
