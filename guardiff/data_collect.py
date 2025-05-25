import os, json, pickle
import numpy as np
from os.path import join
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch

# 1) load diffuser experiment exactly as before
def load_diffusion_manual(logbase, dataset_name, horizon, n_steps, epoch='latest', device='cuda'):
    base = os.path.join(logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')
    cfgs = {}
    for name in ['dataset','render','model','diffusion','trainer']:
        cfgs[name] = pickle.load(open(join(base, f'{name}_config.pkl'),'rb'))
    dataset = cfgs['dataset']()
    renderer = cfgs['render']()
    model   = cfgs['model']().to(device)
    diffusion = cfgs['diffusion'](model).to(device)
    trainer = cfgs['trainer'](diffusion, dataset, renderer)
    if epoch=='latest':
        epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}'))
    trainer.load(epoch)
    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config:  str = 'config.maze2d'

# --------------------------------------
# Setup
# --------------------------------------
args = Parser().parse_args('plan')
env  = datasets.load_environment(args.dataset)
seed = 43

# load diffusion & policy
horizon, T = 384, 256
diff_exp = load_diffusion_manual(args.logbase, args.dataset, horizon, T,
                                 epoch=args.diffusion_epoch, device=args.device)
policy   = Policy(diff_exp.ema, diff_exp.dataset.normalizer)

# prepare logging dirs
traj_root = join('/home/haoran-zhang/Desktop/projects/guardiff', 'trajectories')
os.makedirs(traj_root, exist_ok=True)

# --------------------------------------
# Main rollout & logging loop
# --------------------------------------
episode_id = 0
num_episodes = 1000  # or however many you want to collect

for ep in range(num_episodes):
    obs   = env.reset(seed = seed)
    if args.conditional: env.set_target()
    state = env.state_vector().copy()    # [x, y, vx, vy]

    # conditioning for first plan
    cond = {0: obs}
    target = env._target
    cond[horizon-1] = np.array([*target, 0, 0])

    # get plan once
    action, samples = policy(cond, batch_size=args.batch_size)
    seq_obs   = samples.observations[0]
    seq_acts  = samples.actions[0]

    # storage for this episode
    states, actions, next_states, rewards, dones = [], [], [], [], []

    done = False
    t = 0
    print("start point is",state)
    print('goal point is',target)
    while not done and t < 1600:
        # pick next action from diffusion plan
        
        state = env.state_vector().copy()
        if t < len(seq_obs) - 1:
            next_waypoint = seq_obs[t+1]
        else:
            next_waypoint = seq_obs[-1].copy()
            next_waypoint[2:] = 0

        # step
        a = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        next_obs, r, done, info = env.step(a)
        next_state = env.state_vector().copy()

        # record
        states.append(state)
        actions.append(a)
        next_states.append(next_state)
        rewards.append(r)
        dones.append(done)

        state = next_state
        t += 1

    # success criterion
    success = np.linalg.norm(state[:2] - env._target) < 0.5

    # save to compressed .npz
    fname = join(traj_root,
                 f'ep_{episode_id:04d}_{"succ" if success else "fail"}.npz')
    np.savez_compressed(
        fname,
        state      = np.array(states,      dtype=np.float32),
        action     = np.array(actions,     dtype=np.float32),
        next_state = np.array(next_states, dtype=np.float32),
        reward     = np.array(rewards,     dtype=np.float32),
        done       = np.array(dones,       dtype=np.bool_),
        start      = np.array(state, dtype = np.float32),
        goal       = np.array(target,dtype = np.float32)
    )
    seed =seed + 5
    episode_id += 1
    print(f"Saved episode {episode_id:04d}, success={success}")

