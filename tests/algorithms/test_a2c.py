import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import A2C

from mushroom_rl.policy import GaussianTorchPolicy


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

        nn.init.xavier_uniform_(self._h.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


def learn_a2c():
    mdp = Gym(name='Pendulum-v0', horizon=200, gamma=.99)
    mdp.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    policy_params = dict(
        std_0=1.,
        n_features=64,
        use_cuda=False
    )

    critic_params = dict(network=Network,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    algorithm_params = dict(critic_params=critic_params,
                            actor_optimizer={'class': optim.RMSprop,
                                             'params': {'lr': 7e-4,
                                                        'eps': 3e-3}},
                            max_grad_norm=0.5,
                            ent_coeff=0.01)

    policy = GaussianTorchPolicy(Network,
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
    w_test = np.array([-1.6298926, 1.0359657, -0.34826356, 0.26997435,
                       -0.00908627])

    assert np.allclose(w, w_test)


def test_a2c_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn_a2c()

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in agent_save.__dict__.items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        print('checking ', att)
        print(save_attr, load_attr)

        tu.assert_eq(save_attr, load_attr)
