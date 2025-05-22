import os, json
import numpy as np
from os.path import join
import diffuser.datasets as datasets
import diffuser.utils as utils
from torch.utils.data import Dataset

# --- Setup as before ---
args = Parser().parse_args('plan')
env = datasets.load_environment(args.dataset)

# create directories
data_root = args.savepath + '/trajectories'
os.makedirs(data_root, exist_ok=True)

episode_id = 0
for ep in range(num_episodes):
    obs = env.reset(seed=seed)
    state = env.state_vector().copy()  # [x, y, vx, vy]
    traj = []
    done = False
    t = 0

    while not done and t < env.max_episode_steps:
        action = controller.get_action()
        next_obs, reward, done, info = env.step(action)
        next_state = env.state_vector().copy()
        # record one transition
        traj.append({
            'state':     state.tolist(),
            'action':    action.tolist(),
            'next_state':next_state.tolist(),
            'reward':    float(reward),
            'done':      bool(done)
        })
        state = next_state
        t += 1

    # determine success by reaching within 0.5 of goal
    goal = env._target
    success = np.linalg.norm(state[:2] - goal) < 0.5

    # save episode
    fname = join(data_root, f'episode_{episode_id:04d}_{"succ" if success else "fail"}.npz')
    # stack arrays for compactness
    S = np.array([x['state']      for x in traj], dtype=np.float32)
    A = np.array([x['action']     for x in traj], dtype=np.float32)
    S2= np.array([x['next_state'] for x in traj], dtype=np.float32)
    R = np.array([x['reward']     for x in traj], dtype=np.float32)
    D = np.array([x['done']       for x in traj], dtype=np.bool_)
    np.savez_compressed(fname, state=S, action=A, next_state=S2, reward=R, done=D)
    episode_id += 1