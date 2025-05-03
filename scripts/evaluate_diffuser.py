import os
import pickle
import json
import numpy as np
from os.path import join
import copy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.guides.policies import Policy
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch


def load_diffusion_manual(logbase, dataset_name, horizon, n_steps,
                          epoch='latest', device='cuda'):
    base = os.path.join(logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')
    cfg_names = ['dataset', 'render', 'model', 'diffusion', 'trainer']
    cfgs = {name: pickle.load(open(join(base, f'{name}_config.pkl'), 'rb'))
            for name in cfg_names}

    # instantiate training‐time components
    train_dataset = cfgs['dataset']()
    model         = cfgs['model']().to(device)
    diffusion     = cfgs['diffusion'](model).to(device)
    trainer       = cfgs['trainer'](diffusion, train_dataset, cfgs['render']())

    # load weights
    if epoch == 'latest':
        epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}'))
    trainer.load(epoch)

    return DiffusionExperiment(
        train_dataset,
        cfgs['render'](),
        model,
        diffusion,
        trainer.ema_model,
        trainer,
        epoch
    )

class Parser(utils.Parser):
    dataset:      str = 'maze2d-large-v1'
    config:       str = 'config.maze2d'
    eval_dataset: str = 'maze2d-medium-v1'

if __name__ == '__main__':
    args = Parser().parse_args('plan')

    # 1) build the new eval environment + renderer
    eval_env      = datasets.load_environment(args.eval_dataset)
    eval_renderer = utils.Maze2dRenderer(args.eval_dataset)

    # 2) load the diffusion model (trained on args.dataset)
    h = 384
    T = 256
    diff_exp = load_diffusion_manual(
        args.logbase,
        args.dataset,
        horizon=h,
        n_steps=T,
        epoch=args.diffusion_epoch,
        device=args.device
    )

    policy = Policy(diff_exp.ema, diff_exp.dataset.normalizer)

    # 3) run the rollout in the eval maze
    obs = eval_env.reset(seed=42)
    if args.conditional:
        eval_env.set_target()

    target = eval_env._target
    cond   = {h-1: np.array([*target, 0, 0])}
    rollout = [obs.copy()]
    total_reward = 0

    for t in range(eval_env.max_episode_steps):
        state = eval_env.state_vector().copy()

        if t == 0:
            cond[0] = obs
            action, samples = policy(cond, batch_size=args.batch_size)
            seq = samples.observations[0]
        # follow the sequence of waypoints
        wp = seq[t+1] if t < len(seq)-1 else seq[-1].copy()
        wp[2:] = 0  # zero out velocity at the end
        action = (wp[:2] - state[:2]) + (wp[2:] - state[2:])
        obs, reward, terminal, _ = eval_env.step(action)

        total_reward += reward
        score = eval_env.get_normalized_score(total_reward)
        print(f't={t}  r={reward:.2f}  total={total_reward:.2f}  score={score:.4f}')

        rollout.append(obs.copy())

        # visualize only with the new renderer
        if t == 0:
            eval_renderer.composite(join(args.savepath, 'plan0.png'),
                                    samples.observations, ncol=1)
        if terminal or t % args.vis_freq == 0:
            eval_renderer.composite(join(args.savepath, 'rollout.png'),
                                    np.array(rollout)[None], ncol=1)

        if terminal:
            break

    # 4) save stats
    result = {
        'score': score,
        'step':  t,
        'return': total_reward,
        'term': terminal,
        'epoch_diffusion': diff_exp.epoch
    }
    with open(join(args.savepath, 'rollout.json'), 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f'Done. Results written to {args.savepath}')