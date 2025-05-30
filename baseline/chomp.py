import os
import json
import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from diffuser.datasets import load_environment
import diffuser.utils as utils
import math, random
from environment import maze2d
from environment.maze2d import in_collision 
import time

seed = maze2d.ensure_seed()

def collision_free(env, p1, p2, step_size=0.05):
    """Check if the straight line path from p1 to p2 is collision-free"""
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

def straight_line_path(start, goal, n_steps):
    """Generate a straight line path from start to goal with n_steps waypoints"""
    start = np.array(start)
    goal = np.array(goal)
    path = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0
        point = start + t * (goal - start)
        path.append(point)
    return path

class CHOMPController:
    def __init__(self, env, goal,
                 n_steps=256,
                 lambda_smooth=1.0,
                 lambda_obs=10.0,
                 eta=0.01,
                 max_iterations=100,
                 replan_thresh=1.3,
                 K=50):
        self.env = env
        self.goal = np.array(goal, dtype=np.float32)
        self.n_steps = n_steps
        self.lambda_smooth = lambda_smooth
        self.lambda_obs = lambda_obs
        self.eta = eta
        self.max_iterations = max_iterations
        self.replan_thresh = replan_thresh
        self.K = K
        self.counter = 0
        
        # Initialize path
        start = env.state_vector()[:2].copy()
        self.path = self._optimize_path(start, self.goal)
        
        # Compute velocity path
        self.vel_path = []
        for i in range(len(self.path)-1):
            p0 = np.array(self.path[i])
            p1 = np.array(self.path[i+1])
            self.vel_path.append((p1 - p0) / 0.1)
        self.vel_path.append(np.zeros_like(self.vel_path[0]))
        
        self.ptr = 0
        
    def _optimize_path(self, start, goal):
        """Optimize path using CHOMP algorithm"""
        # Initialize with straight line path
        path = np.array(straight_line_path(start, goal, self.n_steps))
        
        for iteration in range(self.max_iterations):
            # Compute gradient
            gradient = self._compute_gradient(path)
            
            # Update path
            path_new = path - self.eta * gradient
            
            # Keep start and goal fixed
            path_new[0] = start
            path_new[-1] = goal
            
            # Check convergence
            if np.linalg.norm(path_new - path) < 1e-4:
                print(f"CHOMP converged at iteration {iteration}")
                break
                
            path = path_new
            
        return [tuple(p) for p in path]
    
    def _compute_gradient(self, path):
        """Compute the gradient for CHOMP optimization"""
        n = len(path)
        gradient = np.zeros_like(path)
        
        # Smoothness gradient (finite differences)
        for i in range(1, n-1):
            # Second derivative approximation
            gradient[i] += self.lambda_smooth * (2 * path[i] - path[i-1] - path[i+1])
        
        # Obstacle gradient
        for i in range(n):
            obs_grad = self._obstacle_gradient(path[i])
            gradient[i] += self.lambda_obs * obs_grad
            
        return gradient
    
    def _obstacle_gradient(self, point):
        """Compute obstacle gradient at a point"""
        # Simple gradient based on distance to nearest obstacle
        epsilon = 0.1
        grad = np.zeros(2)
        
        # Check collision at current point
        if in_collision(self.env, point):
            # If in collision, push away from obstacles
            # Use finite differences to estimate gradient
            for dim in range(2):
                point_plus = point.copy()
                point_minus = point.copy()
                point_plus[dim] += epsilon
                point_minus[dim] -= epsilon
                
                # Distance to obstacle (negative if inside)
                dist_plus = -1.0 if in_collision(self.env, point_plus) else 1.0
                dist_minus = -1.0 if in_collision(self.env, point_minus) else 1.0
                
                grad[dim] = (dist_plus - dist_minus) / (2 * epsilon)
        else:
            # If not in collision, check nearby points for potential field
            safety_radius = 0.5
            for dim in range(2):
                point_plus = point.copy()
                point_minus = point.copy()
                point_plus[dim] += epsilon
                point_minus[dim] -= epsilon
                
                # Create potential field around obstacles
                dist_plus = self._distance_to_obstacle(point_plus)
                dist_minus = self._distance_to_obstacle(point_minus)
                
                if min(dist_plus, dist_minus) < safety_radius:
                    grad[dim] = (1.0/max(dist_plus, 0.01) - 1.0/max(dist_minus, 0.01)) / (2 * epsilon)
        
        return grad
    
    def _distance_to_obstacle(self, point):
        """Estimate distance to nearest obstacle"""
        if in_collision(self.env, point):
            return 0.0
        
        # Sample points around to estimate distance
        max_dist = 2.0
        for r in np.linspace(0.1, max_dist, 20):
            for angle in np.linspace(0, 2*np.pi, 8):
                test_point = point + r * np.array([np.cos(angle), np.sin(angle)])
                if in_collision(self.env, test_point):
                    return r
        return max_dist
    
    def _replan(self):
        """Replan the path using CHOMP"""
        print("Replanning with CHOMP...")
        start = self.env.state_vector()[:2].copy()
        self.path = self._optimize_path(start, self.goal)
        
        # Recompute velocity path
        self.vel_path = []
        for i in range(len(self.path)-1):
            p0 = np.array(self.path[i])
            p1 = np.array(self.path[i+1])
            self.vel_path.append((p1 - p0) / 0.1)
        self.vel_path.append(np.zeros_like(self.vel_path[0]))
        
        self.ptr = 0
    
    def get_action(self):
        """Get the next action from the CHOMP controller"""
        pos = self.env.state_vector()[:2].copy()
        vel = self.env.state_vector()[2:].copy()
        
        # Check if we need to replan
        idx = min(len(self.path)-1, self.ptr)
        wp = np.array(self.path[idx])
        deviation = np.linalg.norm(pos - wp)
        
        if deviation > self.replan_thresh and self.counter % self.K == 0:
            print(f"Deviation {deviation:.2f} exceeds threshold; replanning")
            self._replan()
            idx = min(len(self.path)-1, self.ptr)
            wp = np.array(self.path[idx])
        
        print("count", self.counter)
        self.counter += 1
        
        # Compute action
        wp_pos = np.array(self.path[idx])
        wp_vel = np.array(self.vel_path[idx])
        action = (wp_pos - pos) + (wp_vel - vel)
        
        # Advance pointer
        self.ptr += 1
        
        return action

def load_diffusion_env(logbase, dataset, horizon, n_steps, device):
    """Load diffusion environment for rendering"""
    base = os.path.join(logbase, dataset, 'diffusion', f'H{horizon}_T{n_steps}')
    cfg_names = ['dataset', 'render', 'model', 'diffusion', 'trainer']
    cfgs = {}
    for name in cfg_names:
        path = os.path.join(base, f'{name}_config.pkl')
        cfgs[name] = pickle.load(open(path, 'rb'))
    
    render = cfgs['render']()
    return render

def main():
    """Main function following the same structure as dynamic_rrt.py"""
    start = time.time()
    
    class Parser(utils.Parser):
        dataset: str = 'maze2d-large-v1'
        config: str = 'config.maze2d'
        horizon: int = 256
        num_particles: int = 128
        sigma: float = 0.3
        lambda_: float = 1.0
        replan_period: int = 100
        replan_thresh: float = 1.3
        vis_freq: int = 50
    
    args = Parser().parse_args('plan')
    
    # Build environment
    def make_env():
        return load_environment(args.dataset)
    
    env = make_env()
    obs = env.reset(seed=seed)
    if args.conditional:
        env.set_target()
    target = env._target
    state = env.state_vector().copy()
    
    # Instantiate CHOMP controller
    chomp = CHOMPController(
        env,
        goal=target,
        n_steps=256,
        lambda_smooth=1.0,
        lambda_obs=10.0,
        eta=0.01,
        max_iterations=100,
        replan_thresh=args.replan_thresh,
        K=50
    )
    
    n_steps = 256
    renderer = load_diffusion_env(
        args.logbase,
        args.dataset,
        horizon=args.horizon,
        n_steps=n_steps,
        device=args.device
    )
    
    rollout = [obs.copy()]
    total_reward = 0.0
    
    for t in range(1600):
        if t == 100:
            print("Interference starts")
            offset = maze2d.teleport_agent(env, level='medium')
            print(f"[t={t}] Teleported by {offset}")
            print("after deviation", env.state_vector())
        
        action = chomp.get_action()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        rollout.append(obs.copy())
        
        # Visualize
        if t % 100 == 0 or done:
            renderer.composite(
                join(args.savepath, f'cur_rollout{t}.png'),
                np.array([rollout]),
                ncol=1
            )
            renderer.composite(
                join(args.savepath, f'cur_chomp_plan{t}.png'),
                np.array([chomp.path]),
                ncol=1
            )
        
        if maze2d.check_done(env):
            print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
            end = time.time()
            break
    
    # Save metrics
    with open(join(args.savepath, 'chomp_rollout.json'), 'w') as f:
        json.dump({
            'steps': len(rollout)-1,
            'return': total_reward,
            'success': bool(done)
        }, f, indent=2)
    
    score = env.get_normalized_score(total_reward)
    end_time = time.time() if 'end' not in locals() else end
    time_taken = end_time - start
    print(f"Elapsed: {time_taken:.4f} s")
    print(f"CHOMP done: steps={len(rollout)-1}, return={total_reward:.2f}, score={score:.2f}")

if __name__ == "__main__":
    main()
