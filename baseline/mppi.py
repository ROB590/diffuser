import os
import json
import pickle
from os.path import join

import numpy as np
import torch
import matplotlib.pyplot as plt

from diffuser.datasets import load_environment
import diffuser.utils as utils
from environment import maze2d
seed = maze2d.ensure_seed()
import time
class MPPIController:
    def __init__(self,
                 master_env,
                 env_fn,
                 horizon: int,
                 num_particles: int,
                 sigma: float,
                 lambda_: float,
                 replan_period: int = 1):
        """
        master_env: the real environment whose state we clone
        env_fn:   zero‐arg callable returning a fresh env clone
        horizon:  planning horizon H
        num_particles: number of sampled rollouts N
        sigma:    std. dev. of Gaussian action noise
        lambda_:  MPPI temperature parameter
        replan_period: only run rollouts every `replan_period` steps
        """
        self.master_env    = master_env
        self.env_fn        = env_fn
        self.horizon       = horizon
        self.N             = num_particles
        self.sigma         = sigma
        self.lambda_       = lambda_
        self.replan_period = replan_period
        self.step_counter  = 0

        # nominal action sequence (H×action_dim)
        self.u_nominal     = np.zeros((horizon, master_env.action_space.shape[0]), dtype=np.float32)

        # create N clones for parallel rollouts
        self.envs          = [env_fn() for _ in range(self.N)]

    def reset(self, start_state: np.ndarray, goal: np.ndarray):
        """Initialize MPPI at the start of an episode."""
        self.start_state  = start_state.copy()
        self.goal         = goal
        self.step_counter = 0
        self.u_nominal[:] = 0  # zero‐initialize action plan

    def step(self, current_state: np.ndarray) -> np.ndarray:
        """
        Return the next action.
        If it's a replanning step, perform full MPPI rollouts; otherwise reuse the existing plan.
        """
        if self.step_counter % self.replan_period == 0:
            # 1) Sample noise & prepare cost buffer
            noise = (np.random.randn(self.N, self.horizon, self.u_nominal.shape[1])
                     .astype(np.float32) * self.sigma)
            costs = np.zeros(self.N, dtype=np.float32)

            # 2) Roll out each particle
            for k, env_k in enumerate(self.envs):
                # Properly sync clone to the master_env's current simulator state
                sim_state = self.master_env.sim.get_state()
                env_k.sim.set_state(sim_state)
                env_k.sim.forward()
                # Copy over the target (Maze2D uses _target)
                env_k._target = self.master_env._target

                total_cost = 0.0
                for t in range(self.horizon):
                    u = self.u_nominal[t] + noise[k, t]
                    _, _, done, _ = env_k.step(u)
                    s = env_k.state_vector()
                    total_cost += np.sum((s[:2] - self.goal[:2])**2)
                    if done:
                        break
                costs[k] = total_cost

            # 3) Debug logs
            print(f"[MPPI] Step {self.step_counter}: noise μ={noise.mean():.4f}, σ={noise.std():.4f}")
            print(f"[MPPI] costs mean={costs.mean():.2f}, min={costs.min():.2f}, max={costs.max():.2f}")

            # 4) Compute importance weights
            cmin = costs.min()
            w   = np.exp(- (costs - cmin) / self.lambda_)
            w  /= w.sum()
            print(f"[MPPI] weights[:5]={w[:5]}, sum={w.sum():.4f}")

            # 5) Update nominal plan
            update = (w[:, None, None] * noise).sum(axis=0)
            self.u_nominal += update

        # 6) Extract and return the first action
        action = self.u_nominal[0].copy()

        # 7) Shift the nominal plan forward
        self.u_nominal[:-1] = self.u_nominal[1:]
        self.u_nominal[-1]  = 0

        self.step_counter += 1
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
    end = None
    class Parser(utils.Parser):
        dataset:       str   = 'maze2d-large-v1'
        config:        str   = 'config.maze2d'
        horizon:       int   = 256
        num_particles: int   = 128
        sigma:         float = 0.3
        lambda_:       float = 1.0
        replan_period: int   = 100
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

    # Instantiate MPPI with proper master_env
    mppi = MPPIController(
        master_env    = env,
        env_fn        = make_env,
        horizon       = args.horizon,
        num_particles = args.num_particles,
        sigma         = args.sigma,
        lambda_       = args.lambda_,
        replan_period = args.replan_period
    )
    n_steps = 256 # FIXME should be args.n_steps
    # renderer = load_diffusion_env( 
    #     args.logbase,
    #     args.dataset,
    #     horizon=args.horizon,
    #     n_steps= n_steps,
    #     device=args.device
    #     )
    # Initialize MPPI’s internal state
    start_state = env.state_vector().copy()
    mppi.reset(start_state, target)

    rollout      = [obs.copy()]
    total_reward = 0.0

    # Main loop
    for t in range(1600):
        state = env.state_vector().copy()
        action = mppi.step(state)
        if t == 100:
            print("Interfer starts")
            offset = maze2d.teleport_agent(env, level='medium')
            print(f"[t={t}] Teleported by {offset}")
            continue
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        rollout.append(obs.copy())

        # Optional: visualize occasionally
        # if t % args.replan_period == 0 :
        #     renderer.composite(
        #         join(args.savepath, f'cur_rollout{t}.png'),
        #          np.array([rollout]),
        #         ncol=1
        #     )

        if maze2d.check_done(env):
            print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
            end = time.time()
            #print(f"Elapsed: {end - start:.4f} s")
            #break

    # Save rollout and metrics
    with open(join(args.savepath, 'mppi_rollout.json'), 'w') as f:
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
    print(f"MPPI done: steps={len(rollout)-1}, return={total_reward:.2f}, score = {score:.2f}")


if __name__ == "__main__":
    main()
