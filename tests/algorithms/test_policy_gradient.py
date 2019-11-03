import numpy as np
import torch

from mushroom.algorithms.policy_search import *
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core import Core
from mushroom.environments.lqr import LQR
from mushroom.policy.gaussian_policy import StateStdGaussianPolicy
from mushroom.utils.parameters import AdaptiveParameter


def learn(alg, alg_params):
    mdp = LQR.generate(dimensions=1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    approximator_params = dict(input_dim=mdp.info.observation_space.shape)
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    sigma = Regressor(LinearApproximator,
                      input_shape=mdp.info.observation_space.shape,
                      output_shape=mdp.info.action_space.shape,
                      params=approximator_params)

    sigma_weights = 2 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    agent = alg(policy, mdp.info, **alg_params)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return policy


def test_REINFORCE():
    params = dict(learning_rate=AdaptiveParameter(value=.01))
    policy = learn(REINFORCE, params)
    w = np.array([-0.0084793 ,  2.00536528])

    assert np.allclose(w, policy.get_weights())


def test_GPOMDP():
    params = dict(learning_rate=AdaptiveParameter(value=.01))
    policy = learn(GPOMDP, params)
    w = np.array([-0.07623939,  2.05232858])

    assert np.allclose(w, policy.get_weights())


def test_eNAC():
    params = dict(learning_rate=AdaptiveParameter(value=.01))
    policy = learn(eNAC, params)
    w = np.array([-0.03668018,  2.05112355])

    assert np.allclose(w, policy.get_weights())
