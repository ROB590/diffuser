
import os
import json
import pickle
from os.path import join
from scipy.interpolate import splprep, splev
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from diffuser.datasets import load_environment
import diffuser.utils as utils
import math, random
from environment import maze2d
from environment.maze2d import in_collision 
import time
seed = maze2d.ensure_seed()
def collision_free2(env, p1, p2, step_size=0.05):
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
def smooth_path(path, is_collision_fn, env, max_iters=100, resolution=0.1):
    smoothed = path.copy()
    for _ in range(max_iters):
        # stop if too few pts remain
        if len(smoothed) < 3:
            break

        # pick two indices at least 2 apart
        i, j = sorted(random.sample(range(len(smoothed)), 2))
        if j - i < 2:
            continue

        p1, p2 = smoothed[i], smoothed[j]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        steps = max(int(dist / resolution), 1)

        # collision check along the straight line
        for k in range(1, steps):
            t = k / steps
            q = (p1[0] + t * dx, p1[1] + t * dy)
            if is_collision_fn(env, *q):
                break
        else:
            # no collision → prune the middle
            smoothed = smoothed[:i+1] + smoothed[j:]
    return smoothed
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
    def collision_free_segment(env, p1, p2, step_size=0.5):
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
        n_steps = int(np.ceil(dist / step_size)) # sometimes start point may contact need replan
        for i in range(1,n_steps + 1):
            q = p1 + vec * (i / n_steps)
            if in_collision(env, q):
                return False
        return True
    tree_s = [Node(*start)]
    tree_g = [Node(*goal)]
    grid = env.unwrapped.maze_arr
    for iter_idx in range(max_iter):
        # 1) Sample random point
        
        rnd = (random.uniform(0, grid.shape[0]),
               random.uniform(0, grid.shape[1]))

        # 2) Extend start‐tree
        near_s = nearest(tree_s, rnd)
        new_s  = steering(near_s, rnd)
        #print('befor collision checking',env.state_vector())
        if in_collision(env, [(new_s.x, new_s.y)]):
            #print('after collision checking',env.state_vector())
            continue
        #print('after collision checking',env.state_vector())
        if not collision_free_segment(env,
                               (near_s.x, near_s.y),
                               (new_s.x, new_s.y),step_size=0.5):
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
                              step_size=extend_len ):
            continue
        tree_g.append(new_g)

        # otherwise, fall back: if direct connect (no collision), accept it
        if not in_collision(env, [(new_g.x, new_g.y)]):
            if not collision_free_segment(env,
                               (new_s.x, new_s.y),
                               (new_g.x, new_g.y),step_size=extend_len):
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
             extend_len=0.1,
             neighbor_radius=0.1,
             goal_radius=0.1,
             vis_every=500):
    """
    Single‐tree RRT* with debug logging and optional visualization.
    """
    # print("maxiter",max_iter)
    # print("extend len",extend_len)
    # print("neighbor_radius",neighbor_radius)
    # print("goal_radius",goal_radius)
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
    #print("rrt called")
    # Initialize
    tree = [Node(start[0], start[1], parent=None, cost=0.0)]
    #print("max_iter",max_iter)
    # print("debug rrt star",env.state_vector())
    # print("does start point collide",in_collision(env,start))
    # print('x limit',env.unwrapped.maze_arr.shape[0])
    # print('y limit',env.unwrapped.maze_arr.shape[1])
    for it in range(1, max_iter+1):
        # — SAMPLE —
        rnd = (
            random.uniform(0, env.unwrapped.maze_arr.shape[0]),
            random.uniform(0, env.unwrapped.maze_arr.shape[1])
        )
        #print("random point is",rnd)
        # — EXTEND —
        near     = nearest(tree, rnd)
        new_node = steering(near, rnd)

        # collision check
        #print('befor collision checking',env.state_vector())
        if in_collision(env, (new_node.x, new_node.y)):
            #print('after collision checking',env.state_vector())
            continue
        #print('after collision checking',env.state_vector())
        # if not collision_free_segment(env,
        #                               (near.x, near.y),
        #                               (new_node.x, new_node.y),
        #                               step_size=extend_len):
        #     print("skip at iter due to line collision",it)
        #     continue

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
                                           step_size=extend_len)):
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
                                           step_size=extend_len)):
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
        #     #print("plotting!!!!!!!!!1")
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
        # print(goal_radius)
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
                 step_size=0.1,
                 radius=0.1,
                 replan_thresh=1.3, # different for small medium and large
                 grow_iters=2000,
                 n_steps=256,K = 50):               # ← new parameter
        self.env           = env
        self.goal          = np.array(goal, dtype=np.float32)
        self.step_size     = step_size
        self.radius        = radius
        self.replan_thresh = replan_thresh
        self.grow_iters    = grow_iters
        self.n_steps       = n_steps     # ← store horizon
        self.counter       = 0
        self.K = K
        self.last_pos = self.env.state_vector()[:2].copy()
        # Initial optimal RRT* plan
        start = env.state_vector()[:2].copy()
        full_path = rrt_star(env,
                             start=start,
                             goal=self.goal,
                             max_iter=self.grow_iters,
                             extend_len=self.step_size,
                             neighbor_radius=0.1,
                             goal_radius=0.1)
        if full_path is None:
            raise RuntimeError("Initial RRT* failed")
        # Resample to exactly n_steps
        self.path = self._resample_path(full_path)
        self.vel_path = []
        for i in range(len(self.path)-1):
            p0 = np.array(self.path[i]); p1 = np.array(self.path[i+1])
            self.vel_path.append((p1 - p0) /0.1)
        # For the final point, assume zero velocity:
        self.vel_path.append(np.zeros_like(self.vel_path[0]))
        #print(f"Initial path length: {len(self.path)} waypoints")
        #print("current path is",self.path)
        self.ptr = 0

    # def _resample_path(self, raw_path):
    #     """ Downsample/pad → shortcut smooth → optional cubic (or lower-order) spline """
    #     # 1) Downsample or pad to self.n_steps
    #     M = len(raw_path)
    #     if M >= self.n_steps:
    #         idxs = np.linspace(0, M - 1, self.n_steps, dtype=int)
    #         pts = [raw_path[i] for i in idxs]
    #     else:
    #         pts = raw_path + [raw_path[-1]] * (self.n_steps - M)

    #     # 2) Shortcut smoothing
    #     def shortcut(pts, iters= max(50, self.n_steps // 4)):
    #         for _ in range(iters):
    #             if len(pts) < 3:
    #                 break
    #             i, j = sorted(random.sample(range(len(pts)), 2))
    #             if j - i < 2:
    #                 continue
    #             if self.collision_free(self.env, pts[i], pts[j]):
    #                 pts = pts[:i+1] + pts[j:]
    #         return pts

    #     pts = shortcut(pts)

    #     # 3) Cubic (or fallback) spline smoothing
    #     xs, ys = zip(*pts)
    #     n_pts = len(xs)

    #     # if too few points, just return pts
    #     if n_pts < 2:
    #         return pts

    #     # choose k based on point count: k <= n_pts - 1, max 3
    #     k = min(3, n_pts - 1)

    #     # splprep needs at least k+1 points
    #     try:
    #         tck, u = splprep([xs, ys], s=0.0, k=k)
    #     except ValueError:
    #         # fallback: skip spline
    #         return pts

    #     # resample uniformly
    #     u_fine = np.linspace(0, 1, self.n_steps)
    #     x_smooth, y_smooth = splev(u_fine, tck)

    #     return list(zip(x_smooth, y_smooth))
    # def _resample_path(self, path):
    #     """Downsample or pad path to exactly self.n_steps waypoints."""
    #     M = len(path)
    #     if M >= self.n_steps:
    #         # pick evenly spaced indices from 0..M-1
    #         idxs = np.linspace(0, M-1, self.n_steps, dtype=int)
    #         path =  [path[i] for i in idxs]
    #     else:
    #         # pad with the final point
    #         #path = path + [path[-1]] * (self.n_steps - M)
    #         # interpolate to upsample from M→n_steps
    #         pts = np.array(path, dtype=float)            # shape (M,2)
    #         xs, ys = pts[:, 0], pts[:, 1]
    #         # target "fractional" indices along original [0, M-1]
    #         idxs_f = np.linspace(0, M - 1, self.n_steps)
    #         xs_new = np.interp(idxs_f, np.arange(M), xs)
    #         ys_new = np.interp(idxs_f, np.arange(M), ys)
    #         path =  list(zip(xs_new.tolist(), ys_new.tolist()))
    #     return path
    def _resample_path(self, path):
        """Downsample or pad path to exactly self.n_steps waypoints."""
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
        #print("replan start point is",start)
        #print("goal",self.goal)
        full_path = rrt_star(self.env,
                             start=start,
                             goal=self.goal,
                             max_iter=self.grow_iters,
                             extend_len=self.step_size,
                             neighbor_radius=0.1,
                             goal_radius=0.1)
        if full_path:
            self.path = self._resample_path(full_path)
            #print("current path is",self.path)
            #print(f"New path length: {len(self.path)} waypoints")
            self.vel_path = []
            for i in range(len(self.path)-1):
                p0 = np.array(self.path[i]); p1 = np.array(self.path[i+1])
                self.vel_path.append((p1 - p0) / 1)
        # For the final point, assume zero velocity:
            self.vel_path.append(np.zeros_like(self.vel_path[0]))
            self.ptr = 0
        else:
            raise RuntimeError('no solution')
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
        new_tail = rrt_star(env,
                             start=start,
                             goal=self.goal,
                             max_iter=self.grow_iters,
                             extend_len=self.step_size,
                             neighbor_radius=0.1,
                             goal_radius=0.1)
        if new_tail is None:
            return False  # repair failed; might need full replan

        # 5) Concatenate prefix + new tail (excluding duplicate start)
        self.path = valid_prefix[:-1] + new_tail
        self.ptr = invalid_idx
        return True
    def get_action(self):
        #print("current path ptr is",self.ptr)
        pos = self.env.state_vector()[:2].copy()
        #print("before action",pos)
        vel = self.env.state_vector()[2:].copy()
        # replan if out of bounds
        # position waypoint
        idx = min(len(self.path)-1,self.ptr) #ignore first 
        wp = np.array(self.path[idx])
        deviation = np.linalg.norm(pos - wp)
        no_replan = True
        # print("before dev",self.ptr)
        # print("bcurrent pos is",pos)
        # print("bcurrent wp is", wp)
        # print("bcurrent path is",self.path)
        # if deviation > self.replan_thresh and self.counter% self.K ==0 and self.counter!=0:
        #     # print("current pos is",pos)
        #     # print("current wp is", wp)
        #     # print("current path is",self.path)
        #     # print(f"Deviation {deviation:.2f} exceeds threshold; replanning")
        #     self._replan()
        #     no_replan = False
        #     # # position waypoint
        #     # print("after dev",self.ptr)
        #     # print("after dev idx",self.ptr)
        #     idx = min(len(self.path)-1,self.ptr)
        #     wp = np.array(self.path[idx]) 
        #     # assert self.ptr == 0
        #     print('larger than thresh')
        #print("deviation",deviation)
        #print(self.replan_thresh)
        # repaired = self._invalidate_and_repair()
        if deviation > self.replan_thresh and self.counter % self.K ==0:
            # fallback full replan once
            print("Local repair needed; replanning")
            self._replan()
            # position waypoint
            idx = min(len(self.path)-1,self.ptr)
            wp = np.array(self.path[idx]) 
            no_replan = False
        self.last_pos = pos.copy()

        print("count",self.counter)
        self.counter = self.counter+1
        # combine into acceleration‐style action
        wp_pos = np.array(self.path[idx])
        wp_vel = np.array(self.vel_path[idx])
        action = (wp_pos - pos) + (wp_vel - vel)
        # print(f"[STEP {self.ptr}] pos={pos}, vel={vel}")               
        # print(f"          wp(idx={idx})={wp}, desired_vel={desired_vel}")
        # print(f"          action={action}")                
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
    start = time.time()
    class Parser(utils.Parser):
        dataset:       str   = 'maze2d-large-v1'
        config:        str   = 'config.maze2d'
        horizon:       int   = 256
        num_particles: int   = 128
        sigma:         float = 0.3
        lambda_:       float = 1.0
        replan_period: int   = 100
        replan_thresh: float = 1.3
        vis_freq:      int   = 50
    args = Parser().parse_args('plan')

    # Build a factory for fresh env clones
    def make_env():
        return load_environment(args.dataset)

    # The “real” env we’ll step through
    env = make_env()
    obs = env.reset(seed=seed)
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
        step_size=0.1,
        radius=0.1,
        replan_thresh=args.replan_thresh,
        grow_iters=400000
    )
    n_steps = 256 # FIXME should be args.n_steps
    # renderer = load_diffusion_env( 
    #     args.logbase,
    #     args.dataset,
    #     horizon=args.horizon,
    #     n_steps= n_steps,
    #     device=args.device
    #     )
    rollout = [obs.copy()]
    total_reward = 0.0

    for t in range(1600): #
        # breakpoint()
        if t == 100:
            print("Interfer starts")
            offset = maze2d.teleport_agent(env, level='medium')
            print(f"[t={t}] Teleported by {offset}")
            print("after deviation",env.state_vector())
        action = drrt.get_action()           # uses dynamic RRT*
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        rollout.append(obs.copy())
        #print("obs",obs)
        # print("t",t)
        # visualize
        # if t % 100 == 0 or done:
            # rollout[0] = [ 2.65957033, 7.55172337  ,1.50555543,-0.14947117]
            # if t!= 0:
            #     rollout[1]=[-4,10,1.50555543,-0.14947117]
            # renderer.composite(
            #     join(args.savepath, f'cur_rollout{t}.png'),
            #      np.array([rollout]),
            #     ncol=1
            # )
            # renderer.composite(
            #     join(args.savepath, f'cur_rrt_plan{t}.png'),
            #      np.array([drrt.path]),
            #     ncol=1
            # )
            #print('drrt path',drrt.path)
            #print('drrt vpath',drrt.vel_path)
            # waypoints = drrt.path[:t]
            # rollout_pts = rollout

            # # 1) align lengths
            # T = min(len(waypoints), len(rollout_pts))

            # # 2) build truncated series
            # times  = list(range(T))
            # wp_pts = waypoints[:T]
            # ro_pts = rollout_pts[:T]

            # wp_x = [p[0] for p in wp_pts]
            # wp_y = [p[1] for p in wp_pts]
            # ro_x = [p[0] for p in ro_pts]
            # ro_y = [p[1] for p in ro_pts]

            # # 3) plotting
            # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6))

            # ax1.plot(times, wp_x, '-',  label='waypoint $x$', linestyle='--')
            # ax1.plot(times, ro_x, '-o', label='rollout $x$',  ms=4)
            # ax1.set_ylabel('x position')
            # ax1.legend()

            # ax2.plot(times, wp_y, '-',  label='waypoint $y$', linestyle='--')
            # ax2.plot(times, ro_y, '-o', label='rollout $y$',  ms=4)
            # ax2.set_ylabel('y position')
            # ax2.set_xlabel('time step')
            # ax2.legend()

            # plt.tight_layout()
            # plt.show()
        if maze2d.check_done(env):
            print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
            end = time.time()
            #break

    # save metrics
    with open(join(args.savepath, 'drrt_rollout.json'),'w') as f:
        json.dump({
            'steps': len(rollout)-1,
            'return': total_reward,
            'success': bool(done)
        }, f, indent=2)
    score = env.get_normalized_score(total_reward)
    if end == None:
        end = time.time()
    time_taken = start - end
    print(f"Elapsed: {time_taken:.4f} s")
    print(f"Dynamic RRT* done: steps={len(rollout)-1}, return={total_reward:.2f},score = {score:.2f}")

if __name__ == "__main__":
    main()