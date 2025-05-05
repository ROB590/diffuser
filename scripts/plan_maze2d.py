import json
import numpy as np
from os.path import join
import os
import pdb
import pickle
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.models import TemporalUnet, GaussianDiffusion
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch
####################### For loading and testing with different horizon#########
def load_diffusion_manual(logbase, dataset_name, horizon, n_steps, epoch='latest', device='cuda'):
    base = os.path.join(logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')
    # 1. Load the five core configs as before:
    cfg_names = ['dataset', 'render', 'model', 'diffusion', 'trainer']
    cfgs = {}
    for name in cfg_names:
        path = os.path.join(base, f'{name}_config.pkl')
        cfgs[name] = pickle.load(open(path, 'rb'))           

    # 2. Instantiate each component:
    dataset = cfgs['dataset']()
    renderer = cfgs['render']()
    model   = cfgs['model']().to(device)
    diffusion = cfgs['diffusion'](model).to(device)
    trainer = cfgs['trainer'](diffusion, dataset, renderer)

    # 3. Load weights:
    if epoch == 'latest':
        epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}'))
    trainer.load(epoch)

    # 4. Return the complete experiment object:
    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

#logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)  

#---------------------------------- loading ----------------------------------#

# diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

# diffusion = diffusion_experiment.ema
# dataset = diffusion_experiment.dataset
# renderer = diffusion_experiment.renderer

# policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- custom loading ----------------------------------#

h=384
T=256
diffusion_exp = load_diffusion_manual(
    args.logbase,
    args.dataset,
    horizon=h,
    n_steps=T,
    epoch=args.diffusion_epoch,
    device=args.device
)
diffusion = diffusion_exp.ema
dataset = diffusion_exp.dataset
renderer = diffusion_exp.renderer
policy   = Policy(diffusion_exp.ema, diffusion_exp.dataset.normalizer)
print("Evaluating with horizon: ", h,"n_steps: ", T,"on datasets with horizon", args.horizon, "n_steps: ", args.n_diffusion_steps)
#---------------------------------- main loop ----------------------------------#

observation = env.reset(seed = 42)

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation

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



    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        # print(
        #     f'maze | pos: {xy} | goal: {goal}'
        # )

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

#logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_exp.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

# import json
# import numpy as np
# from os.path import join
# import pdb

# from diffuser.guides.policies import Policy
# import diffuser.datasets as datasets
# import diffuser.utils as utils


# class Parser(utils.Parser):
#     dataset: str = 'maze2d-umaze-v1'
#     config: str = 'config.maze2d'

# #---------------------------------- setup ----------------------------------#

# args = Parser().parse_args('plan')

# # logger = utils.Logger(args)

# env = datasets.load_environment(args.dataset)

# #---------------------------------- loading ----------------------------------#

# diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

# diffusion = diffusion_experiment.ema
# dataset = diffusion_experiment.dataset
# renderer = diffusion_experiment.renderer

# policy = Policy(diffusion, dataset.normalizer)

# #---------------------------------- main loop ----------------------------------#

# observation = env.reset()

# if args.conditional:
#     print('Resetting target')
#     env.set_target()

# ## set conditioning xy position to be the goal
# target = env._target
# cond = {
#     diffusion.horizon - 1: np.array([*target, 0, 0]),
# }

# ## observations for rendering
# rollout = [observation.copy()]

# total_reward = 0
# for t in range(env.max_episode_steps):

#     state = env.state_vector().copy()

#     ## can replan if desired, but the open-loop plans are good enough for maze2d
#     ## that we really only need to plan once
#     if t == 0:
#         cond[0] = observation

#         action, samples = policy(cond, batch_size=args.batch_size)
#         actions = samples.actions[0]
#         sequence = samples.observations[0]
#     # pdb.set_trace()

#     # ####
#     if t < len(sequence) - 1:
#         next_waypoint = sequence[t+1]
#     else:
#         next_waypoint = sequence[-1].copy()
#         next_waypoint[2:] = 0
#         # pdb.set_trace()

#     ## can use actions or define a simple controller based on state predictions
#     action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
#     # pdb.set_trace()
#     ####

#     # else:
#     #     actions = actions[1:]
#     #     if len(actions) > 1:
#     #         action = actions[0]
#     #     else:
#     #         # action = np.zeros(2)
#     #         action = -state[2:]
#     #         pdb.set_trace()



#     next_observation, reward, terminal, _ = env.step(action)
#     total_reward += reward
#     score = env.get_normalized_score(total_reward)
#     print(
#         f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
#         f'{action}'
#     )

#     if 'maze2d' in args.dataset:
#         xy = next_observation[:2]
#         goal = env.unwrapped._target
#         print(
#             f'maze | pos: {xy} | goal: {goal}'
#         )

#     ## update rollout observations
#     rollout.append(next_observation.copy())

#     # logger.log(score=score, step=t)

#     if t % args.vis_freq == 0 or terminal:
#         fullpath = join(args.savepath, f'{t}.png')

#         if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


#         # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

#         ## save rollout thus far
#         renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

#         # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

#         # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

#     if terminal:
#         break

#     observation = next_observation

# # logger.finish(t, env.max_episode_steps, score=score, value=0)

# ## save result as a json file
# json_path = join(args.savepath, 'rollout.json')
# json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}
# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)