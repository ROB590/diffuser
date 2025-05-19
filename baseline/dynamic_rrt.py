
import os
import json
import pickle
from os.path import join

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from diffuser.datasets import load_environment
import diffuser.utils as utils
import math, random

def in_collision(env, q=None, ground_names=None):
    """
    Return True if point q=[x,y] collides with non-ground geoms.
    """
    sim_state = env.sim.get_state()
    if q is not None:
        env.sim.data.qpos[:2] = np.array(q, dtype=np.float32)
    env.sim.forward()

    ncon = env.sim.data.ncon
    contacts = env.sim.data.contact[:ncon]
    if ground_names is None:
        ground_names = {'floor','ground','wall_floor'}
    for c in contacts:
        g1 = env.sim.model.geom_id2name(c.geom1)
        g2 = env.sim.model.geom_id2name(c.geom2)
        if g1 in ground_names or g2 in ground_names:
            continue
        env.sim.set_state(sim_state)
        env.sim.forward()
        return True

    env.sim.set_state(sim_state)
    env.sim.forward()
    return False


def rrt_connect(env,start, goal,
                max_iter=500, extend_len=1.0, horizon=256,
                ):
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
    def nearest(tree, pt):
        return min(tree, key=lambda n: (n.x - pt[0])**2 + (n.y - pt[1])**2)

    def build_path(node):
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    def collision_free_segment(env, p1, p2, step_size=0.1):
        """
        Returns True iff the straight line path from p1→p2
        never collides (according to in_collision) when sampled
        at intervals of length <= step_size.
        """
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist == 0:
            return not in_collision(env, p1)
        n_steps = int(np.ceil(dist / step_size))
        for i in range(n_steps + 1):
            q = p1 + vec * (i / n_steps)
            if in_collision(env, q):
                return False
        return True

    tree_s = [Node(*start)]
    tree_g = [Node(*goal)]
    grid = env.unwrapped.maze_arr
    for iter_idx in range(max_iter):
        # 1) Sample random point
        
        rnd = (random.uniform(0, grid.shape[1]),
               random.uniform(0, grid.shape[0]))

        # 2) Extend start‐tree
        near_s = nearest(tree_s, rnd)
        new_s  = steering(near_s, rnd)
        if in_collision(env, [(new_s.x, new_s.y)]):
            continue
        if not collision_free_segment(env,
                               (near_s.x, near_s.y),
                               (new_s.x, new_s.y),step_size=extend_len / 2):
            continue
        tree_s.append(new_s)

        # 3) Try connect goal‐tree toward new_s
        near_g = nearest(tree_g, (new_s.x, new_s.y))
        new_g  = steering(near_g, (new_s.x, new_s.y))
        if in_collision(env, [(new_g.x, new_g.y)]):
            continue
        if not collision_free_segment(env,
                              (near_s.x, near_s.y),
                              (new_s.x, new_s.y),
                              step_size=extend_len / 2):
            continue
        tree_g.append(new_g)

        # otherwise, fall back: if direct connect (no collision), accept it
        if not in_collision(env, [(new_g.x, new_g.y)]):
            if not collision_free_segment(env,
                               (new_s.x, new_s.y),
                               (new_g.x, new_g.y),step_size=extend_len / 2):
                continue
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
def rrt_star(env, start, goal,
             max_iter=10000,
             extend_len=0.2,
             neighbor_radius=0.2,
             goal_radius=0.2,
             vis_every=1000):
    """
    Single‐tree RRT* with debug logging and optional visualization.
    """

    # 1) Node now carries a cost‐to‐come
    class Node:
        __slots__ = ('x','y','parent','cost')
        def __init__(self, x, y, parent=None, cost=0.0):
            self.x = x; self.y = y
            self.parent = parent
            self.cost   = cost

    # 2) Steering exactly as before
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

    # 3) Brute‐force nearest
    def nearest(tree, pt):
        return min(tree, key=lambda n: (n.x - pt[0])**2 + (n.y - pt[1])**2)

    # 4) Reconstruct path
    def build_path(node):
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    # 5) Segment‐collision sampling
    def collision_free_segment(env, p1, p2, step_size=0.1):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist == 0:
            return not in_collision(env, p1)
        steps = int(np.ceil(dist / step_size))
        for i in range(steps+1):
            q = p1 + vec * (i/steps)
            if in_collision(env, q):
                return False
        return True

    # Initialize
    tree = [Node(start[0], start[1], parent=None, cost=0.0)]

    for it in range(1, max_iter+1):
        # — SAMPLE —
        rnd = (
            random.uniform(0, env.unwrapped.maze_arr.shape[1]),
            random.uniform(0, env.unwrapped.maze_arr.shape[0])
        )

        # — EXTEND —
        near     = nearest(tree, rnd)
        new_node = steering(near, rnd)

        # collision check
        if in_collision(env, (new_node.x, new_node.y)):
            continue
        if not collision_free_segment(env,
                                      (near.x, near.y),
                                      (new_node.x, new_node.y),
                                      step_size=extend_len/2):
            continue

        # — GATHER NEIGHBORS —
        nbrs = [
            (idx, node) for idx,node in enumerate(tree)
            if math.hypot(node.x - new_node.x, node.y - new_node.y)
               <= neighbor_radius
        ]

        # Debug: print neighbor count
        #print(f"[iter {it:3d}] neighbors found: {len(nbrs)}")

        # — CHOOSE BEST PARENT —
        best_cost   = near.cost + math.hypot(near.x - new_node.x,
                                             near.y - new_node.y)
        best_parent = near
        for idx, nbr in nbrs:
            c_through = nbr.cost + math.hypot(nbr.x - new_node.x,
                                              nbr.y - new_node.y)
            if (c_through < best_cost
                and collision_free_segment(env,
                                           (nbr.x, nbr.y),
                                           (new_node.x, new_node.y),
                                           step_size=extend_len/2)):
                best_cost   = c_through
                best_parent = nbr

        new_node.parent = best_parent
        new_node.cost   = best_cost
        tree.append(new_node)

        # — REWIRE —
        for idx, nbr in nbrs:
            c_via_new = new_node.cost + math.hypot(nbr.x - new_node.x,
                                                   nbr.y - new_node.y)
            if (c_via_new < nbr.cost
                and collision_free_segment(env,
                                           (new_node.x, new_node.y),
                                           (nbr.x, nbr.y),
                                           step_size=extend_len/2)):
                nbr.parent = new_node
                nbr.cost   = c_via_new

        # — LOG BEST DISTANCE TO GOAL —
        best_dist = min(
            math.hypot(n.x - goal[0], n.y - goal[1])
            for n in tree
        )
        print(f"[iter {it:3d}] tree size={len(tree):4d}, best_goal_dist={best_dist:.3f}")

        # # — OPTIONAL VISUALIZATION —
        # if it % vis_every == 0 or best_dist <= goal_radius:
        #     xs = [n.x for n in tree]
        #     ys = [n.y for n in tree]
        #     plt.figure(figsize=(4,4))
        #     plt.scatter(xs, ys, s=5, alpha=0.5)
        #     for n in tree:
        #         if n.parent is not None:
        #             plt.plot([n.x, n.parent.x],
        #                      [n.y, n.parent.y],
        #                      color='gray', alpha=0.3)
        #     plt.scatter(start[0], start[1], c='green', s=50, label='start')
        #     plt.scatter(goal[0], goal[1], c='red',   s=50, label='goal')
        #     plt.title(f"iter {it}, best_dist={best_dist:.3f}")
        #     plt.legend(); plt.show()

        # — GOAL CHECK —
        if best_dist <= goal_radius:
            # find the node closest to goal
            goal_node = min(tree,
                            key=lambda n: math.hypot(n.x-goal[0], n.y-goal[1]))
            return build_path(goal_node)

    # no path found
    print("RRT* failed: no path within max_iter")
    return None
class DynamicRRTStarController:
    def __init__(self, env, goal,
                 step_size=0.2,
                 radius=0.5,
                 replan_thresh=1,
                 grow_iters=2000,
                 n_steps=256):               # ← new parameter
        self.env           = env
        self.goal          = np.array(goal, dtype=np.float32)
        self.step_size     = step_size
        self.radius        = radius
        self.replan_thresh = replan_thresh
        self.grow_iters    = grow_iters
        self.n_steps       = n_steps     # ← store horizon

        # Initial optimal RRT* plan
        start = env.state_vector()[:2].copy()
        full_path = rrt_star(env,
                             start=start,
                             goal=self.goal,
                             max_iter=self.grow_iters,
                             extend_len=self.step_size,
                             neighbor_radius=self.radius,
                             goal_radius=self.step_size)
        if full_path is None:
            raise RuntimeError("Initial RRT* failed")
        # Resample to exactly n_steps
        self.path = self._resample_path(full_path)
        print(f"Initial path length: {len(self.path)} waypoints")
        print("current path is",self.path)
        self.ptr = 0

    def _resample_path(self, path):
        """Downsample or pad `path` to exactly self.n_steps waypoints."""
        M = len(path)
        if M >= self.n_steps:
            # pick evenly spaced indices from 0..M-1
            idxs = np.linspace(0, M-1, self.n_steps, dtype=int)
            return [path[i] for i in idxs]
        else:
            # pad with the final point
            return path + [path[-1]] * (self.n_steps - M)

    def _replan(self):
        print("Replanning RRT*…")
        start = self.env.state_vector()[:2].copy()
        full_path = rrt_star(self.env,
                             start=start,
                             goal=self.goal,
                             max_iter=self.grow_iters,
                             extend_len=self.step_size,
                             neighbor_radius=self.radius,
                             goal_radius=self.step_size)
        if full_path:
            self.path = self._resample_path(full_path)
            print("current path is",self.path)
            print(f"New path length: {len(self.path)} waypoints")
            self.ptr = 0
        return
    def collision_free(self,env, p1, p2, step_size=0.1):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist == 0:
            return not in_collision(env, p1)
        steps = int(np.ceil(dist / step_size))
        for i in range(steps+1):
            q = p1 + vec * (i/steps)
            if in_collision(env, q):
                return False
        return True
    def _invalidate_and_repair(self):
        # 1) Identify invalid waypoints (collision on segment ahead)
        invalid_idx = None
        for i in range(self.ptr, len(self.path)-1):
            if not self.collision_free(self.env,
                                          self.path[i],
                                          self.path[i+1],
                                            ):
                invalid_idx = i
                break

        # 2) If nothing invalid, no replan needed
        if invalid_idx is None:
            return False  # no repair

        # 3) Prune waypoints from invalid_idx onward
        valid_prefix = self.path[:invalid_idx+1]

        # 4) Regrow RRT* from current position to goal using grow_iters
        start = np.array(valid_prefix[-1], dtype=np.float32)
        new_tail = rrt_star(self.env,
                            start=start,
                            goal=self.goal,
                            max_iter=self.grow_iters,
                            extend_len=self.step_size,
                            neighbor_radius=self.radius,
                            goal_radius=self.step_size)
        if new_tail is None:
            return False  # repair failed; might need full replan

        # 5) Concatenate prefix + new tail (excluding duplicate start)
        self.path = valid_prefix[:-1] + new_tail
        self.ptr = invalid_idx
        return True
    def get_action(self):
        pos = self.env.state_vector()[:2].copy()
        vel = self.env.state_vector()[2:]
        # replan if out of bounds
        repaired = self._invalidate_and_repair()
        if not repaired and self.ptr >= len(self.path):
            # fallback full replan once
            self._replan() 

        # position waypoint
        idx = min(len(self.path)-1,self.ptr)
        wp = np.array(self.path[idx])
        # compute desired velocity from next waypoint
        if self.ptr + 1 < len(self.path):
            wp_next = np.array(self.path[self.ptr+1])
            desired_vel = wp_next - wp
        else:
            desired_vel = np.zeros_like(wp)

        # combine into acceleration‐style action
        action = (wp - pos) + (desired_vel - vel)
        print(f"[STEP {self.ptr}] pos={pos}, vel={vel}")               # :contentReference[oaicite:4]{index=4}
        print(f"          wp(idx={idx})={wp}, desired_vel={desired_vel}")# :contentReference[oaicite:5]{index=5}
        print(f"          action={action}")                 # :contentReference[oaicite:6]{index=6}
        # advance to next waypoint when close enough
        # if np.linalg.norm(pos - wp) < 0.2:
        #     print(f"        → reached wp[{self.ptr}], advancing")      
        #     self.ptr += 1
        # else:
        #     print("not reach due to error",np.linalg.norm(pos - wp))
        self.ptr +=1
        return action

def load_diffusion_env(logbase, dataset, horizon, n_steps, device):
    # only to get target logic; no diffusion used
    base = os.path.join(logbase, dataset, 'diffusion', f'H{horizon}_T{n_steps}')
    cfg_names = ['dataset', 'render', 'model', 'diffusion', 'trainer']
    cfgs = {}
    for name in cfg_names:
        path = os.path.join(base, f'{name}_config.pkl')
        cfgs[name] = pickle.load(open(path, 'rb'))

    # env     = cfgs['dataset']()
    render    = cfgs['render']()
    return render
def main():
     # Parse arguments (uses diffuser.utils.Parser for consistency)
    class Parser(utils.Parser):
        dataset:       str   = 'maze2d-large-v1'
        config:        str   = 'config.maze2d'
        horizon:       int   = 256
        num_particles: int   = 128
        sigma:         float = 0.3
        lambda_:       float = 1.0
        replan_period: int   = 100
        replan_thresh: float = 1
        vis_freq:      int   = 50
    args = Parser().parse_args('plan')

    # Build a factory for fresh env clones
    def make_env():
        return load_environment(args.dataset)

    # The “real” env we’ll step through
    env = make_env()
    obs = env.reset(seed=42)
    if args.conditional:
        env.set_target()
    target = env._target
    state = env.state_vector().copy()
    #breakpoint()
    # path = rrt_star(env, start = state[:2],goal= target)
    # print('start point is',state[:2])
    # print('end point is',target)
    # print('path',path)
    # Instantiate the Dynamic RRT* controller
    drrt = DynamicRRTStarController(
        env,
        goal=target,
        step_size=0.2,
        radius=0.05,
        replan_thresh=args.replan_thresh,
        grow_iters=200000
    )
    n_steps = 256 # FIXME should be args.n_steps
    renderer = load_diffusion_env( 
        args.logbase,
        args.dataset,
        horizon=args.horizon,
        n_steps= n_steps,
        device=args.device
        )
    rollout = [obs.copy()]
    total_reward = 0.0

    for t in range(env.max_episode_steps):
        action = drrt.get_action()           # uses dynamic RRT*
        # breakpoint()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        rollout.append(obs.copy())

        # visualize
        if t % 50 == 0 or done:
            renderer.composite(
                join(args.savepath, f'cur_rollout{t}.png'),
                 np.array([rollout]),
                ncol=1
            )
        if done:
            print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
            break

    # save metrics
    with open(join(args.savepath, 'drrt_rollout.json'),'w') as f:
        json.dump({
            'steps': len(rollout)-1,
            'return': total_reward,
            'success': bool(done)
        }, f, indent=2)
    score = env.get_normalized_score(total_reward)
    print(f"Dynamic RRT* done: steps={len(rollout)-1}, return={total_reward:.2f},score = {score:.2f}")

if __name__ == "__main__":
    main()