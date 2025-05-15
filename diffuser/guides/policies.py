# from collections import namedtuple
# # import numpy as np
# import torch
# import einops
# import pdb
# import numpy as np
# import diffuser.utils as utils
# # from diffusion.datasets.preprocessing import get_policy_preprocess_fn

# Trajectories = namedtuple('Trajectories', 'actions observations')
# # GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

# class Policy:

#     def __init__(self, diffusion_model, normalizer):
#         self.diffusion_model = diffusion_model
#         self.normalizer = normalizer
#         self.action_dim = normalizer.action_dim

#     @property
#     def device(self):
#         parameters = list(self.diffusion_model.parameters())
#         return parameters[0].device

#     def _format_conditions(self, conditions, batch_size):
#         conditions = utils.apply_dict(
#             self.normalizer.normalize,
#             conditions,
#             'observations',
#         )
#         conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
#         conditions = utils.apply_dict(
#             einops.repeat,
#             conditions,
#             'd -> repeat d', repeat=batch_size,
#         )
#         return conditions

#     def __call__(self, conditions, debug=False, batch_size=1):


#         conditions = self._format_conditions(conditions, batch_size)

#         ## batchify and move to tensor [ batch_size x observation_dim ]
#         # observation_np = observation_np[None].repeat(batch_size, axis=0)
#         # observation = utils.to_torch(observation_np, device=self.device)

#         ## run reverse diffusion process
#         sample = self.diffusion_model(conditions)
#         sample = utils.to_np(sample)

#         # ## extract action [ batch_size x horizon x transition_dim ]
#         # actions = sample[:, :, :self.action_dim]
#         # actions = self.normalizer.unnormalize(actions, 'actions')
#         # # actions = np.tanh(actions)

#         # ## extract first action
#         # action = actions[0, 0]
#         if self.action_dim > 0:
#             normed_actions = sample[:, :, :self.action_dim]
#             actions = self.normalizer.unnormalize(normed_actions, 'actions')
#             action = actions[0, 0]
#         else:
#             # no actions: set empty array and a default zero action
#             actions = np.zeros((batch_size, sample.shape[1], 0), dtype=np.float32)
#             action = np.zeros((0,), dtype=np.float32)

#         # if debug:
#         normed_observations = sample[:, :, self.action_dim:]
#         observations = self.normalizer.unnormalize(normed_observations, 'observations')

#         # if deltas.shape[-1] < observation.shape[-1]:
#         #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
#         #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
#         #     deltas = np.concatenate([deltas, padding], axis=-1)

#         # ## [ batch_size x horizon x observation_dim ]
#         # next_observations = observation_np + deltas.cumsum(axis=1)
#         # ## [ batch_size x (horizon + 1) x observation_dim ]
#         # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

#         trajectories = Trajectories(actions, observations)
#         return action, trajectories
#         # else:
#         #     return action
from collections import namedtuple
import torch
import einops
import numpy as np
import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')

class Policy:
    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer      = normalizer
        self.horizon         = diffusion_model.horizon
        self.action_dim      = normalizer.action_dim
        self.state_dim       = normalizer.observation_dim  # dim of observations

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def _format_conditions(self, conditions, batch_size):
        cond = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations'
        )
        cond = utils.to_torch(cond, dtype=torch.float32, device=self.device)
        cond = utils.apply_dict(
            einops.repeat,
            cond,
            'd -> repeat d', repeat=batch_size
        )
        return cond

    def __call__(self,
        conditions,              # dict {time: state}
        batch_size      = 1,
        diffusion_steps = None,  # # steps to run (None ⇒ full chain)
        replan_mode     = 'scratch',  # 'scratch' or 'future'
        prefix_states   = None       # np.array [batch, t+1, state_dim] when future
    ):
        # 1) normalize & expand your conditioning
        conds = self._format_conditions(conditions, batch_size)

        # 2) build initial latent x
        if replan_mode == 'scratch' or prefix_states is None:
            # full replan: pure noise
            x = torch.randn(
                (batch_size, self.horizon, self.state_dim),
                device=self.device
            )
            diffusion_steps = None
        else:
            # future replan: keep prefix, noise the tail
            t = prefix_states.shape[1]
            # normalize prefix
            prefix_norm = self.normalizer.normalize(prefix_states, 'observations')
            prefix_t   = utils.to_torch(prefix_norm, dtype=torch.float32, device=self.device)
            # build x = [prefix_t | noise]
            noise = torch.randn((batch_size, self.horizon, self.state_dim), device=self.device)
            noise[:, :t, :] = prefix_t
            x = noise

        # 3) run exactly `diffusion_steps` reverse steps
        #    conditional_sample will call p_sample_loop under the hood
        # breakpoint()
        samples = self.diffusion_model.conditional_sample(
            cond=conds,
            samples=x,
            diffusion_step=diffusion_steps
        )

        # 4) back to numpy
        sample = utils.to_np(samples)

        # 5) extract actions
        normed_actions = sample[:, :, :self.action_dim]
        #actions        = self.normalizer.unnormalize(normed_actions, 'actions')
        if self.action_dim > 0:
            actions = self.normalizer.unnormalize(normed_actions, 'actions')
            action  = actions[0, 0]
        else:
            # no actions: produce an empty array (or zeros) and a default action
            actions = np.zeros((batch_size, sample.shape[1], 0), dtype=np.float32)
            action  = np.zeros((0,), dtype=np.float32)

        # 6) extract observations
        normed_obs   = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_obs, 'observations')

        return action, Trajectories(actions, observations)
