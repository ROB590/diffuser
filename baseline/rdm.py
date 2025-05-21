import os
import json
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.models import TemporalUnet, GaussianDiffusion
from diffuser.utils.serialization import DiffusionExperiment, get_latest_epoch
import torch
from environment import maze2d
seed = maze2d.ensure_seed()
#######################
# Helper functions
#######################
#fast replan helper
def fast_replan(policy, diffusion_model, cond, sequence, plan_ptr, batch_size=1, diffusion_steps=100):  
    """  
    Implements fast replanning by combining current conditions with previous trajectory  
    and running a reduced number of diffusion steps.  
    """  
    device = next(diffusion_model.parameters()).device  
    sequence_tensor = torch.tensor(sequence).float().unsqueeze(0).to(device)  
      
    current_state = torch.tensor(cond[0]).float().unsqueeze(0).to(device)  
    prefix_length = plan_ptr + 1  
    prefix_length = min(prefix_length, len(sequence))    
    prefix_states = sequence[:prefix_length]  
    prefix_states[-1] = cond[0] 
    prefix_np = np.expand_dims(prefix_states, axis=0)  

    _, samples = policy(  
        cond,  
        batch_size=batch_size,  
        diffusion_steps=diffusion_steps,  
        replan_mode='future',  
        prefix_states=prefix_np  
    )  
      
    return samples
def plot_loglikelihood(ts, L_vals, save_dir):
    """
    Plot and save the log‐likelihood (KL) curve.
    Args:
        ts (List[int]): timesteps at which we measured L_t
        L_vals (List[float]): measured average KL values
        save_dir (str): directory to save the plot into
    """
    plt.figure(figsize=(6,4))
    plt.plot(ts, L_vals, '-o', linewidth=2, markersize=4)
    plt.xlabel('Time step t')
    plt.ylabel('Average KL (L_t)')
    plt.title('Adaptive Replanning: Log‐Likelihood over Time')
    plt.grid(True)
    out_path = os.path.join(save_dir, 'loglikelihood_vs_t.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f" → saved loglikelihood plot to {out_path}")
def load_diffusion_manual(logbase, dataset_name, horizon, n_steps, epoch='latest', device='cuda'):
    base = os.path.join(logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')
    cfg_names = ['dataset', 'render', 'model', 'diffusion', 'trainer']
    cfgs = {}
    for name in cfg_names:
        path = os.path.join(base, f'{name}_config.pkl')
        cfgs[name] = pickle.load(open(path, 'rb'))

    dataset_obj = cfgs['dataset']()
    renderer    = cfgs['render']()
    model       = cfgs['model']().to(device)
    diffusion   = cfgs['diffusion'](model).to(device)
    trainer     = cfgs['trainer'](diffusion, dataset_obj, renderer)

    if epoch == 'latest':
        #epoch = get_latest_epoch((logbase, dataset_name, 'diffusion', f'H{horizon}_T{n_steps}')) #
        #FIXME hard code checkpoints
        epoch = 980000 #200000
        print("Current horizon",horizon)
        print('current step',n_steps)
    trainer.load(epoch)

    return DiffusionExperiment(
        dataset_obj, renderer, model, diffusion, trainer.ema_model, trainer, epoch
    )

#######################
# Replanning determinator
#######################
# 1) Hyperparameters for adaptive replanning
ls       = 0.5       # full‐replan threshold (tune on validation)
lf       = 0.7          # partial‐replan threshold (ls < lf)
I        = [50,100,125,175,200]#[5,10,15] # diffusion steps to sample for KL estimate #NOTE must smaller than the diffusion noise step # 50,100,125,175,200
ts = []       # list of timesteps
L_vals = []   # corresponding log‐likelihood values
horizon = 256   #trajectory length
n_steps = 256   #diffusion steps
# 2) Decision function
def should_replan(diffusion, old_seq,rollout, cond, t, ls, lf, I):
    device = next(diffusion.parameters()).device
    # 2.1 Build partial trajectory tau0 with real observations
    tau0 = old_seq.copy()
    for k in range(0, t+1):
        tau0[k] = rollout[k]  # replace with actual observed state
    plan = np.expand_dims(tau0, axis=0)
    renderer.composite(
        join(args.savepath, f'ood_trajectory_m_t{t}.png'),
        plan,
        ncol=1
    )
    assert tau0.shape[0]==horizon
    # 2.2 Estimate average KL over selected timesteps
    kl_vals = []
    for i in I:
        idx = torch.full((1,), i, dtype=torch.long, device=device)
        x0 = torch.tensor(tau0).float().unsqueeze(0).to(device)
        # forward noise
        noise = torch.randn_like(x0)
        x_i = diffusion.q_sample(x0, idx, noise=noise)
        # true posterior
        μ_true, _, logvar_true = diffusion.q_posterior(x0, x_i, idx)
        # model posterior
        μ_pred, var_pred, logvar_pred = diffusion.p_mean_variance(x_i, cond, idx)
        # KL divergence per Eq. (11)
        kl = 0.5*((logvar_pred - logvar_true)
                  + (torch.exp(logvar_true - logvar_pred))
                  + (μ_true - μ_pred).pow(2)*torch.exp(-logvar_pred)
                  - 1).mean()
        kl_vals.append(kl.item())
    L_t = sum(kl_vals) / len(kl_vals)

    # 2.3 Apply thresholds
    print(f"Average loglikelihood at {t}, is {L_t}")
    if L_t <= ls:
        return 'scratch',L_t
    elif L_t <= lf:
        return 'future',L_t
    else:
        return 'none',L_t


#######################
# Argument parsing
#######################
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config:  str = 'config.maze2d'

args = Parser().parse_args('plan')


#######################
# Environment & Policy setup
#######################
env = datasets.load_environment(args.dataset)

# Custom load of diffusion (to match your horizons)
diff_exp = load_diffusion_manual(
    args.logbase,
    args.dataset,
    horizon=horizon,
    n_steps=n_steps,
    epoch=args.diffusion_epoch,
    device=args.device
)
diffusion = diff_exp.ema
dataset   = diff_exp.dataset
renderer  = diff_exp.renderer
policy    = Policy(diffusion, dataset.normalizer)
Ns = args.horizon # NOTE should match to horizon
Nf = 80 #NOTE self define steps for future replan
print(f"Evaluating with horizon={horizon}, n_steps={n_steps}")

#######################
# Main control loop
#######################
observation  = env.reset(seed = seed) #42
state        = env.state_vector().copy()
if args.conditional:
    env.set_target()
target       = env._target

K            = 100     # replan every K steps
rollout      = [observation.copy()]
total_reward = 0.0
sequence     = None
plan_ptr     = 0
L_t = 0      # temp holder
global_history = rollout.copy()
for t in range(env.max_episode_steps): #env.max_episode_steps
    state = env.state_vector().copy()

    # 1) init plan
    if t == 0:
        print(f"[t={t}] Init from start {state[:2]} to target {target}")

        # build conditioning dict
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
        #breakpoint()
        _, samples = policy(cond, batch_size=args.batch_size,diffusion_steps = Ns,replan_mode = 'scratch')
        sequence   = samples.observations[0]   # (horizon, state_dim)
        plan_ptr   = 0
        # also save your composite if desired
        renderer.composite(
            join(args.savepath, f'plan_{t}.png'),
            samples.observations,
            ncol=1
        )
    if t < diffusion.horizon and (t % K) == 0:
        mode,L_t = should_replan(diffusion, sequence,global_history, cond, plan_ptr, ls, lf, I)
    else:
        mode = None
    if mode == 'scratch':
        print(f"[t={t}] Replanning from start {state[:2]} to target {target} by scratch")
        print(f"[t={t}] Full replanning; resetting rollout")
        global_history = [env.state_vector().copy()]
        # full replanning (Algorithm 2)
        # build conditioning dict
        cond = {
            0:                   state.copy(),
            diffusion.horizon-1: np.array([*target, 0, 0])
        }
        _, samples = policy(cond, batch_size=args.batch_size,diffusion_steps = Ns,replan_mode = 'scratch')
        sequence = samples.observations[0]
        # visualize the scratch replan
        rrtplan = np.expand_dims(sequence, axis=0)
        renderer.composite(
            join(args.savepath, f'plan_scratch_t{t}.png'),
            rrtplan,
            ncol=1
        )
        plan_ptr = 0
    elif mode == 'future':  
        print(f"[t={t}] Partial replanning (future)")  
        
        # Use the modified fast_replan function  
        samples_fut = fast_replan(  
            policy=policy,  
            diffusion_model=diffusion,  
            cond=cond,  
            sequence=sequence,  
            plan_ptr=plan_ptr,  
            batch_size=args.batch_size,  
            diffusion_steps=Nf  
        )  
        
        sequence = samples_fut.observations[0]  
        
        # Visualize and log  
        print("current plan ptr is", plan_ptr)  
        print("current state", state.copy())  
        print("next way point is", sequence[plan_ptr])  
        
        fplan = np.expand_dims(sequence, axis=0)  
        renderer.composite(  
            join(args.savepath, f'plan_future_t{t}.png'),  
            fplan,  
            ncol=1  
        )
        # plan_ptr = 0
    next_waypoint  = sequence[plan_ptr]        # [x,y,vx,vy]

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    if t == 100:
        print("Interfer starts")
        offset = maze2d.teleport_agent(env, level='largetr=')
        print(f"[t={t}] Teleported by {offset}")
        continue
    next_obs, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    rollout.append(next_obs.copy())
    global_history.append(next_obs.copy())
    # 4) Advance the pointer AFTER stepping
    plan_ptr = min(plan_ptr + 1, len(sequence)-1)
    ts.append(t)
    L_vals.append(L_t)
    # 5) Check for termination # FIXME not actually works
    if terminal:
        print(f"🏁 Terminated at step {t}, return={total_reward:.2f}")
        break
    # for debuging (output current planning trajectory vs current rollouts)
    if t% K ==0:
        current_waypoint = np.expand_dims(sequence, axis=0)   # sequence is the H×obs_dim future‐patched plan
        renderer.composite(
                join(args.savepath, f'cur_wpt{t}.png'),
                current_waypoint,
                ncol=1
            )
        renderer.composite(
                join(args.savepath, f'cur_rollout{t}.png'),
                 np.array([rollout]),
                ncol=1
            )
plot_loglikelihood(ts, L_vals, args.savepath)
# 6) Final dump
renderer.composite(
    join(args.savepath, 'final_rollout.png'),
    np.array([rollout]),
    ncol=1
)
with open(join(args.savepath, 'rollout.json'), 'w') as f:
    json.dump({
        'step':  t,
        'return': total_reward,
        'term':   terminal,
        'score':  env.get_normalized_score(total_reward)
    }, f, indent=2)

print(f"Done. Steps={len(rollout)-1}, Return={total_reward:.2f},Score = {score:.2f}")
