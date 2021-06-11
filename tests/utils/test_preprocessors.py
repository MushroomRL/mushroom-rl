import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.utils.parameters import Parameter

from mushroom_rl.policy import EpsGreedy

from mushroom_rl.algorithms.value import DQN

from mushroom_rl.core import Core

from mushroom_rl.approximators.parametric import TorchApproximator
from torch import optim, nn

from mushroom_rl.environments import Gym
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor


def test_normalizing_preprocessor(tmpdir):
    np.random.seed(88)

    class Network(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()

            n_input = input_shape[-1]
            n_output = output_shape[0]

            self._h1 = nn.Linear(n_input, n_output)

            nn.init.xavier_uniform_(self._h1.weight,
                                    gain=nn.init.calculate_gain('relu'))

        def forward(self, state, action=None):
            q = F.relu(self._h1(torch.squeeze(state, 1).float()))
            if action is None:
                return q
            else:
                action = action.long()
                q_acted = torch.squeeze(q.gather(1, action))
                return q_acted

    mdp = Gym('CartPole-v0', horizon=500, gamma=.99)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    input_shape = mdp.info.observation_space.shape

    approximator_params = dict(network=Network,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               n_features=2, use_cuda=False)

    alg_params = dict(batch_size=5, initial_replay_size=10,
                      max_replay_size=500, target_update_frequency=50)

    agent = DQN(mdp.info, pi, TorchApproximator,
                approximator_params=approximator_params, **alg_params)

    norm_box = MinMaxPreprocessor(mdp_info=mdp.info,
                                  clip_obs=5.0, alpha=0.001)

    core = Core(agent, mdp, preprocessors=[norm_box])

    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    # training correctly
    assert (core._state.min() >= -norm_box._clip_obs
            and core._state.max() <= norm_box._clip_obs)

    # loading and setting data correctly
    state_dict1 = norm_box.get_state()
    norm_box.save(tmpdir / 'norm_box.msh')

    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    norm_box = MinMaxPreprocessor.load(tmpdir / 'norm_box.msh')
    state_dict2 = norm_box.get_state()

    assert ((state_dict1["mean"] == state_dict2["mean"]).all()
            and (state_dict1["var"] == state_dict2["var"]).all()
            and state_dict1["count"] == state_dict2["count"])

    core = Core(agent, mdp, preprocessors=[norm_box])
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

