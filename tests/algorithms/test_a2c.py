import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.algorithms.actor_critic import A2C

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork


def learn_a2c():
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    policy_params = dict(std_0=1., n_features=None, n_layers=0)

    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    algorithm_params = dict(critic_params=critic_params,
                            actor_optimizer={'class': optim.RMSprop,
                                             'params': {'lr': 7e-4,
                                                        'eps': 3e-3}},
                            max_grad_norm=0.5,
                            ent_coeff=0.01)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = A2C(mdp.info, policy, **algorithm_params)

    core = Core(agent, mdp)
    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent


def test_a2c():
    agent = learn_a2c()

    w = agent.policy.get_weights()
    w_test = np.array([0.662464, -1.3380364, -0.1384504, -0.00665062])

    assert np.allclose(w, w_test)


def test_a2c_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn_a2c()

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        print('checking ', att)
        print(save_attr, load_attr)

        tu.assert_eq(save_attr, load_attr)
