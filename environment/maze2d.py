#helper function for experiment
import numpy as np
import os
import torch
import random
#Ensure reproductivity
def within_bounds(pos, maze_arr):
    """
    pos: (x, y) tuple or array
    maze_arr: numpy array giving the occupancy grid, shape = (height, width)
    """
    x, y = pos
    height, width = maze_arr.shape
    return (0 <= x < height-1) and (0 <= y < width-1)

def ensure_seed():
    seed = 42
    os.environ['PYTHONHASHSEED']            = str(seed)
    random.seed(seed)                       
    np.random.seed(seed)                    
    torch.manual_seed(seed)                 
    torch.cuda.manual_seed(seed)            
    torch.cuda.manual_seed_all(seed)        
    torch.backends.cudnn.deterministic      = True
    torch.backends.cudnn.benchmark          = False
    #torch.use_deterministic_algorithms(True)
    return seed

def step_with_action_noise(env, action, level):
    # Noise std fractions
    sigma_frac = {'small':0.1, 'medium':0.3, 'large':0.5}[level]
    sigma = sigma_frac * env.action_space.high
    noisy_action = action + np.random.normal(0, sigma)
    print("original_action",action,"nosiy_action",noisy_action)
    return env.step(np.clip(noisy_action, -env.action_space.high, env.action_space.high))

def teleport_agent(env, level: str,seed = 42,max_attempts = 100):
    # TODO must with in the environment
    np.random.seed(seed)   
    """
    Teleports the Maze2D agent to a different (x,y) position.
    level: 'small', 'medium', or 'large'
    """
    original_state = env.unwrapped.sim.get_state()
    frac_map = {'small': 0.1, 'medium': 0.3, 'large': 0.6}
    f = frac_map[level]
    max_extent = 3 # np.max(env.observation_space.high[:2])
    for _ in range(max_attempts):
        # sample random offset
        offset = (np.random.rand(2)*2 - 1) * f * max_extent

        # apply to a copy of the saved state
        sim_state = original_state
        sim_state.qpos[:2] += offset
        env.unwrapped.sim.set_state(sim_state)
        env.unwrapped.sim.forward()
        if not in_collision(env,sim_state.qpos[:2]) and within_bounds(sim_state.qpos[:2], env.unwrapped.maze_arr):
            print("debug tele",env.state_vector())
            print("Not in collision apply interfer")
            return offset  # found collision-free teleport
        env.unwrapped.sim.set_state(original_state)
        env.unwrapped.sim.forward()

    # 6) fallback: no valid teleport
    env.unwrapped.sim.set_state(original_state)
    env.unwrapped.sim.forward()
    return np.zeros(2)

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

def check_done(env,tol = 0.15):  #from reward logic of maze2d
    return np.linalg.norm(env.state_vector()[:2]- env._target) < tol