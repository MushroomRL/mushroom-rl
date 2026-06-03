import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.rl_utils.parameters import Parameter

from mushroom_rl.policy import EpsGreedy

from mushroom_rl.algorithms.value import DQN

from mushroom_rl.core import Core

from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork
from torch import optim

from mushroom_rl.environments import Gymnasium
from mushroom_rl.rl_utils.preprocessors import MinMaxPreprocessor

from copy import deepcopy


def test_normalizing_preprocessor(tmpdir):
    np.random.seed(88)

    mdp = Gymnasium('CartPole-v1', horizon=500, gamma=.99)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Approximator
    input_shape = mdp.info.observation_space.shape

    approximator_params = dict(network=QNetwork,
                               optimizer={'class':  optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               n_features=None
                               )

    alg_params = dict(batch_size=5, initial_replay_size=10,
                      max_replay_size=500, target_update_frequency=50)

    agent = DQN(mdp.info, pi, NumpyTorchApproximator, approximator_params=approximator_params, **alg_params)

    norm_box = MinMaxPreprocessor(mdp_info=mdp.info, clip_obs=5.0, alpha=0.001)
    agent.add_core_preprocessor(norm_box)

    core = Core(agent, mdp)

    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)
    dataset = core.evaluate(n_steps=1000)

    # training correctly
    assert (dataset.state.min() >= -norm_box._clip_obs and dataset.state.max() <= norm_box._clip_obs)

    # save current dict
    state_dict1 = deepcopy(norm_box.__dict__)

    # save preprocessor and agent
    norm_box.save(tmpdir / 'norm_box.msh')
    agent.save(tmpdir / 'agent.msh')

    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    norm_box_new = MinMaxPreprocessor.load(tmpdir / 'norm_box.msh')

    agent_new = DQN.load(tmpdir / 'agent.msh')

    assert len(agent_new.core_preprocessors) == 1

    norm_box_agent = agent_new.core_preprocessors[0]

    state_dict2 = norm_box_new.__dict__
    state_dict3 = norm_box_agent.__dict__

    assert (state_dict1["_obs_runstand"].mean == state_dict2["_obs_runstand"].mean).all() \
           and (state_dict1["_obs_runstand"].std == state_dict2["_obs_runstand"].std).all()

    assert (state_dict1["_obs_runstand"].mean == state_dict3["_obs_runstand"].mean).all() \
           and (state_dict1["_obs_runstand"].std == state_dict3["_obs_runstand"].std).all()


def test_normalizing_preprocessor_backend():
    # check if the preprocessor work the same for numpy and torch
    np.random.seed(88)

    mdp = Gymnasium('CartPole-v1', horizon=500, gamma=.99)

    mdp_info_torch = deepcopy(mdp.info)
    mdp_info_torch.backend = 'torch'

    norm_box_np = MinMaxPreprocessor(mdp_info=mdp.info, clip_obs=5.0, alpha=0.001)
    norm_box_torch = MinMaxPreprocessor(mdp_info=mdp_info_torch, clip_obs=5.0, alpha=0.001)

    mdp.reset()
    for i in range(20):
        action = np.random.randint(1, size=mdp.info.action_space.shape)
        next_state, _, absorbing, _ = mdp.step(action)

        next_state_np = norm_box_np(next_state)
        next_state_torch = norm_box_torch(torch.from_numpy(next_state)).detach().cpu().numpy()

        assert np.allclose(next_state_np, next_state_torch, rtol=1e-4)

        norm_box_np.update(next_state)
        norm_box_torch.update(torch.from_numpy(next_state))

        if absorbing:
            mdp.reset()
