import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.rl_utils.parameters import Parameter

from mushroom_rl.policy import EpsGreedy

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.core import Agent

from mushroom_rl.core import Core

from mushroom_rl.approximators.parametric.networks import QNetwork
from torch import optim

from mushroom_rl.environments import Gymnasium
from mushroom_rl.rl_utils.preprocessors import MinMaxPreprocessor

from copy import deepcopy


def test_normalizing_preprocessor(tmpdir):
    np.random.seed(42)
    torch.manual_seed(42)

    mdp = Gymnasium('CartPole-v1', horizon=500, gamma=.99)
    mdp.seed(42)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random, backend='torch')

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

    agent = DQN(mdp.info, pi, approximator_params=approximator_params, **alg_params)

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
    np.random.seed(42)

    mdp = Gymnasium('CartPole-v1', horizon=500, gamma=.99)

    mdp_info_torch = deepcopy(mdp.info)
    mdp_info_torch.backend = 'torch'

    norm_box_np = MinMaxPreprocessor(mdp_info=mdp.info, clip_obs=5.0, alpha=0.001)
    norm_box_torch = MinMaxPreprocessor(mdp_info=mdp_info_torch, clip_obs=5.0, alpha=0.001)

    mdp.seed(42)
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


def test_minmax_preprocessor_does_not_mutate_its_input():
    from mushroom_rl.core import MDPInfo
    from mushroom_rl.core.spaces import Box

    bounded = MDPInfo(Box(np.zeros(3), 2 * np.ones(3)), Box(-np.ones(1), np.ones(1)), .99, 100)
    partly_unbounded = MDPInfo(Box(np.array([0., -np.inf, 0.]), np.array([2., np.inf, 2.])),
                               Box(-np.ones(1), np.ones(1)), .99, 100)

    for mdp_info in (bounded, partly_unbounded):
        preprocessor = MinMaxPreprocessor(mdp_info=mdp_info)
        obs = np.array([[0.5, 1.5, 0.25]])
        original = obs.copy()

        normalized = preprocessor(obs)

        assert np.array_equal(obs, original)
        assert not np.array_equal(normalized, original)


def test_agent_preprocessor_with_history_keeps_flat_statistics(tmpdir):
    from mushroom_rl.algorithms.actor_critic import PPO
    from mushroom_rl.approximators.parametric.networks import FeedForwardNetwork, ActorNetwork
    from mushroom_rl.environments import InvertedPendulum
    from mushroom_rl.policy import GaussianTorchPolicy
    from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor

    np.random.seed(1)
    torch.manual_seed(1)

    mdp = InvertedPendulum(horizon=50)
    history_length = 4
    window_shape = (history_length,) + mdp.info.observation_space.shape

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
                         loss=F.mse_loss, n_features=None, n_layers=0,
                         input_shape=window_shape, output_shape=(1,))

    policy = GaussianTorchPolicy(ActorNetwork, window_shape, mdp.info.action_space.shape,
                                 std_0=1., n_features=None, n_layers=0)

    agent = PPO(mdp.info, policy, actor_optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
                n_epochs_policy=2, batch_size=32, eps_ppo=.2, lam=.95,
                critic_params=critic_params, history_length=history_length)

    preprocessor = StandardizationPreprocessor(mdp.info, backend='torch')
    agent.add_agent_preprocessor(preprocessor)

    core = Core(agent, mdp)
    core.learn(n_episodes=2, n_episodes_per_fit=1, quiet=True)

    assert preprocessor._obs_runstand.mean.shape == mdp.info.observation_space.shape
    assert preprocessor._obs_runstand.std.shape == mdp.info.observation_space.shape
    assert preprocessor._obs_runstand._n == 101

    dataset = core.evaluate(n_episodes=1, quiet=True)
    assert len(dataset) == 50

    agent.save(tmpdir / 'agent_history_preprocessor.msh')
    agent_new = Agent.load(tmpdir / 'agent_history_preprocessor.msh')

    assert len(agent_new.history_manager.preprocessors) == 1
    loaded = agent_new.history_manager.preprocessors[0]
    assert np.allclose(loaded._obs_runstand.mean, preprocessor._obs_runstand.mean)
    assert np.allclose(loaded._obs_runstand.std, preprocessor._obs_runstand.std)

    agent_new.add_agent_preprocessor(StandardizationPreprocessor(mdp.info, backend='torch'))
    assert len(agent_new.history_manager.preprocessors) == 2


def test_minmax_preprocessor_multi_dimensional_observation():
    from mushroom_rl.core import MDPInfo
    from mushroom_rl.core.spaces import Box

    mdp_info = MDPInfo(Box(np.zeros((2, 2)), 2 * np.ones((2, 2))), Box(-np.ones(1), np.ones(1)), .99, 100)
    preprocessor = MinMaxPreprocessor(mdp_info=mdp_info)

    assert preprocessor._obs_mask.shape == (2, 2)
    assert not preprocessor._run_norm_obs

    normalized = preprocessor(np.array([[[0., 1.], [2., 0.5]]]))

    assert np.allclose(normalized, np.array([[[-1., 0.], [1., -0.5]]]))


def test_minmax_preprocessor_single_bounded_component():
    from mushroom_rl.core import MDPInfo
    from mushroom_rl.core.spaces import Box

    low = np.array([-np.inf, 0., -np.inf])
    high = np.array([np.inf, 4., np.inf])
    mdp_info = MDPInfo(Box(low, high), Box(-np.ones(1), np.ones(1)), .99, 100)
    preprocessor = MinMaxPreprocessor(mdp_info=mdp_info)

    assert preprocessor._run_norm_obs
    assert preprocessor._obs_mask.sum() == 1

    normalized = preprocessor(np.array([[1., 1., 1.]]))

    assert np.isclose(normalized[0, 1], -0.5)
