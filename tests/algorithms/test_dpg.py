import numpy as np
import torch
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic import COPDAC_Q
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.rl_utils.parameters import Parameter


def learn_copdac_q():
    n_steps = 50
    mdp = InvertedPendulum(horizon=n_steps)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Agent
    n_tilings = 1
    alpha_theta = Parameter(5e-3 / n_tilings)
    alpha_omega = Parameter(0.5 / n_tilings)
    alpha_v = Parameter(0.5 / n_tilings)
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings=tilings)

    input_shape = (phi.size,)

    mu = Regressor(LinearApproximator, input_shape=input_shape, output_shape=mdp.info.action_space.shape, phi=phi)

    sigma = 1e-1 * np.eye(1)
    policy = GaussianPolicy(mu, sigma)

    agent = COPDAC_Q(mdp.info, policy, mu, alpha_theta, alpha_omega, alpha_v, value_function_features=phi)

    # Train
    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)
    
    return agent


def test_copdac_q():
    policy = learn_copdac_q().policy
    w = policy.get_weights()
    w_test = np.array([[0.0, -6.62180045e-07, 0.0, -4.23972882e-02]])

    assert np.allclose(w, w_test)


def test_copdac_q_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn_copdac_q()

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
