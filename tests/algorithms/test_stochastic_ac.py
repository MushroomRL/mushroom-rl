import numpy as np
import torch
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic import StochasticAC, StochasticAC_AVG
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import StateLogStdGaussianPolicy
from mushroom_rl.rl_utils.parameters import Parameter


def learn(alg):
    n_steps = 50
    mdp = InvertedPendulum(horizon=n_steps)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Agent
    n_tilings = 2
    alpha_r = Parameter(.0001)
    alpha_theta = Parameter(.001 / n_tilings)
    alpha_v = Parameter(.1 / n_tilings)
    tilings = Tiles.generate(n_tilings-1, [1, 1],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings=tilings)

    tilings_v = tilings + Tiles.generate(1, [1, 1],
                                         mdp.info.observation_space.low,
                                         mdp.info.observation_space.high + 1e-3)
    psi = Features(tilings=tilings_v)

    input_shape = (phi.size,)

    mu = Regressor(LinearApproximator, input_shape=input_shape, output_shape=mdp.info.action_space.shape, phi=phi)

    std = Regressor(LinearApproximator, input_shape=input_shape, output_shape=mdp.info.action_space.shape, phi=phi)

    std_0 = np.sqrt(1.)
    std.set_weights(np.log(std_0) / n_tilings * np.ones(std.weights_size))

    policy = StateLogStdGaussianPolicy(mu, std)

    if alg is StochasticAC:
        agent = alg(mdp.info, policy, alpha_theta, alpha_v, lambda_par=.5,  value_function_features=psi)
    elif alg is StochasticAC_AVG:
        agent = alg(mdp.info, policy, alpha_theta, alpha_v, alpha_r, lambda_par=.5, value_function_features=psi)
    else:
        assert False

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return agent


def test_stochastic_ac():
    policy = learn(StochasticAC).policy

    w = policy.get_weights()
    w_test = np.array([-0.0026135, 0.01222979])

    assert np.allclose(w, w_test)


def test_stochastic_ac_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(StochasticAC)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_stochastic_ac_avg():
    policy = learn(StochasticAC_AVG).policy

    w = policy.get_weights()
    w_test = np.array([-0.00295433, 0.01325534])

    assert np.allclose(w, w_test)


def test_stochastic_ac_avg_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(StochasticAC_AVG)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
