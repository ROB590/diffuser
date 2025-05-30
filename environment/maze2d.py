#helper function for experiment
import numpy as np
import os
import torch
import random
import gym
from gym.envs.registration import register

import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

#Ensure reproductivity
def within_bounds(pos, maze_arr):
    """
    pos: (x, y) tuple or array
    maze_arr: numpy array giving the occupancy grid, shape = (height, width)
    """
    x, y = pos
    height, width = maze_arr.shape
    return (0 <= x < height-1) and (0 <= y < width-1)

def ensure_seed(seed = 42):
    seed = seed
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

# External disturbance
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


# Environment Changes function
class ExternalDisturbanceWrapper(gym.Wrapper):
    """
    Two‐mode disturbance over each window:
      • Teleport (‘teleop’) exactly teleop_count times
      • Action‐noise bursts of length 20 steps, N bursts per window
    After window_size steps, resample schedules.
    Flags in info: 'teleop_disturbance', 'action_disturbance'
    """
    def __init__(
        self,
        env,
        disturb_type: str = 'both',            # 'teleop', 'action', or 'both'
        window_size: int = 400,
        teleop_count: int = 2,
        action_burst_count: int = 10,          # how many bursts in a window
        burst_duration: int = 30,              # length of each burst
        noise_level: float = 0.8
    ):
        super().__init__(env)
        assert disturb_type in ('teleop','action','both')
        self.disturb_type       = disturb_type
        self.window_size        = window_size
        self.teleop_count       = teleop_count
        self.action_burst_count = action_burst_count
        self.burst_duration     = burst_duration
        self.noise_level        = noise_level

        self.step_idx           = 0
        self.teleop_steps       = []
        self.burst_starts       = []
        self.current_noise      = None
        self.steps_left_in_burst= 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.step_idx = 0
        idxs = np.arange(self.window_size)

        # schedule teleops
        if self.disturb_type in ('teleop','both'):
            self.teleop_steps = list(np.random.choice(
                idxs, size=self.teleop_count, replace=False
            ))
        else:
            self.teleop_steps = []

        # schedule action‐noise bursts
        if self.disturb_type in ('action','both'):
            self.burst_starts = list(np.random.choice(
                idxs, size=self.action_burst_count, replace=False
            ))
        else:
            self.burst_starts = []

        # clear any ongoing burst
        self.current_noise       = None
        self.steps_left_in_burst = 0

        return obs

    def step(self, action):
        info = {}
        do_teleop = self.step_idx in self.teleop_steps
        do_action = self.step_idx in self.burst_starts

        # Teleport disturbance
        if do_teleop:
            offset = teleport_agent(self.env, level='medium')
            info['teleop_disturbance'] = offset

        # Start a new noise burst if we hit a burst start
        if do_action:
            self.steps_left_in_burst = self.burst_duration
            sigma = self.noise_level * self.action_space.high
            # sample one fixed noise vector
            self.current_noise = np.random.randn(*action.shape) * sigma

        # If we’re in an active burst, apply the same noise
        if self.steps_left_in_burst > 0:
            action = action + self.current_noise
            action = np.clip(action,
                             -self.action_space.high,
                              self.action_space.high)
            info['action_disturbance'] = True
            self.steps_left_in_burst -= 1

        obs, reward, done, env_info = self.env.step(action)

        # advance and wrap window
        self.step_idx = (self.step_idx + 1) % self.window_size

        env_info.update(info)
        return obs, reward, done, env_info

    @property
    def _target(self):
        return self.env._target

class StartEndRandomWrapper(gym.Wrapper):
    """
    On reset(), picks a random collision‐free start and goal within the maze.
    Flags 'start_randomized' and 'goal_randomized' in info.
    """
    def __init__(self, env, max_attempts=1000):
        super().__init__(env)
        self.max_attempts = max_attempts

    def _sample_free(self):
        maze = self.unwrapped.maze_arr
        H, W = maze.shape
        for _ in range(self.max_attempts):
            x = np.random.uniform(0, H)
            y = np.random.uniform(0, W)
            pos = np.array([x, y], dtype=np.float32)
            if within_bounds(pos, maze) and not in_collision(self.env, pos):
                return pos
        raise RuntimeError("Failed to sample a collision-free point")

    def reset(self, **kwargs):
        # 1) Sample a new start position
        new_start = self._sample_free()
        state = self.unwrapped.sim.get_state()
        state.qpos[:2] = new_start
        self.unwrapped.sim.set_state(state)
        self.unwrapped.sim.forward()

        # 2) Sample a new goal position
        new_goal = self._sample_free()
        self.env._target = new_goal.copy()

        # 3) Perform underlying reset (to get correct obs)
        obs = self.env.reset(**kwargs)
        self.new_start = new_start
        self.new_goal = new_goal
        # 4) Return obs and info flags
        info = {
            'start_randomized': new_start,
            'goal_randomized':  new_goal
        }
        return obs

    def step(self, action):
        # Pass through, no change during episode
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    @property
    def _target(self):
        return self.env._target




############### Modified environments
# Import utility functions from the original renderer
def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def plot2img(fig, remove_margins=True):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

# Custom maze definitions
LARGE_MAZE_TRAP = \
    "############\\" + \
    "#OOOO#OOOOO#\\" + \
    "#O##O#O#O#O#\\" + \
    "#OOOOOO#OOO#\\" + \
    "#O#O#OOO##O#\\" + \
    "#OO#O#OOOOO#\\" + \
    "##O#O#O#O###\\" + \
    "#OO#OOO#OGO#\\" + \
    "############"

LARGER_MAZE = \
    "############\\" + \
    "#OOOOOOOOOO#\\" + \
    "#OOOO#OOOOO#\\" + \
    "#O##O#O#O#O#\\" + \
    "#OOOOOO#OOO#\\" + \
    "#O####O###O#\\" + \
    "#OO#O#OOOOO#\\" + \
    "##O#O#O#G###\\" + \
    "#OO#OOO#OOO#\\" + \
    "#OOOOOOOOOO#\\" + \
    "############"

def load_custom_env(env_name, max_episode_steps=1000):
    """Load custom maze environment based on env_name."""
    name = None
    
    if env_name == 'maze-editv0-large':
        name = 'MazeEdit-v0'
        register(
            id=name,
            entry_point='d4rl.pointmaze.maze_model:MazeEnv',
            kwargs={
                'maze_spec': LARGE_MAZE_TRAP,
                'reward_type': 'dense',
                'reset_target': False,
            }
        )
        
    
    elif env_name == 'maze-editv1-large':
        name = 'MazeEdit-v1'
        register(
            id=name,
            entry_point='d4rl.pointmaze.maze_model:MazeEnv',
            kwargs={
                'maze_spec': LARGER_MAZE,
                'reward_type': 'dense',
                'reset_target': False,
            }
        )
    
    assert name is not None, f"Unknown environment: {env_name}"
    
    env = gym.make(name)
    env = env.unwrapped
    env.max_episode_steps = max_episode_steps
    env.name = name
    
    return env


MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12),
    'maze-editv0-large':(0,9,0,12),
    'maze-editv1-large':(0,11,0,12)
}

class MazeRenderer:

    def __init__(self, env):
        if type(env) is str: env = load_custom_env(env)
        self._config = env._config
        self._background = self._config != ' '
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

class Maze2dRenderer(MazeRenderer):

    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_custom_env(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        observations = observations + .5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)
