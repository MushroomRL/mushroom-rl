import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic import DDPG, TD3
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork


class RecordingPreprocessor:
    def __init__(self, scale):
        self.scale = scale
        self.shapes = list()

    def __call__(self, state):
        self.shapes.append(tuple(state.shape))

        return state * self.scale


def learn(alg, setup=None):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    initial_replay_size = 100
    max_replay_size = 5000
    batch_size = 200
    tau = .001

    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=None,
                        n_layers=0,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    agent = alg(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    if setup is not None:
        setup(agent)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    assert agent._fit_count > 0

    return agent


def test_ddpg():
    policy = learn(DDPG).policy
    w = policy.get_weights()
    w_test = torch.tensor([-0.00764989, 1.23425770, 0.10049075])

    assert torch.allclose(w, w_test)


def test_ddpg_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(DDPG)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_td3():
    policy = learn(TD3).policy
    w = policy.get_weights()
    w_test = torch.tensor([1.34045994, 0.93246555, 0.19654009])

    assert torch.allclose(w, w_test)


def test_td3_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(TD3)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_ddpg_fit_preprocessor_identity():
    preprocessor = RecordingPreprocessor(1.)
    policy = learn(DDPG, lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()
    w_test = torch.tensor([-0.00764989, 1.23425770, 0.10049075])

    assert torch.allclose(w, w_test)
    assert preprocessor.shapes == [(200, 2)] * 4


def test_ddpg_fit_preprocessor():
    preprocessor = RecordingPreprocessor(0.)
    policy = learn(DDPG, lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()
    w_test = torch.tensor([-0.00564953, 1.23625839, 0.10049096])

    assert torch.allclose(w, w_test)
    assert preprocessor.shapes == [(200, 2)] * 4


def test_td3_fit_preprocessor():
    preprocessor = RecordingPreprocessor(0.)
    policy = learn(TD3, lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()
    w_test = torch.tensor([1.34145999, 0.93346554, 0.19654009])

    assert torch.allclose(w, w_test)
    assert preprocessor.shapes == [(200, 2)] * 4


def test_ddpg_overwrite_actor_parameters():
    parameters = dict()

    def exclude_bias(agent):
        parameters['default'] = agent.actor_parameters
        weight, bias = agent.actor_parameters
        parameters['weight'], parameters['bias'] = weight.detach().clone(), bias.detach().clone()
        agent.overwrite_actor_parameters([weight])

    agent = learn(DDPG, exclude_bias)

    weight, bias = agent.policy._approximator.parameters()

    assert len(parameters['default']) == 2
    assert parameters['default'][0] is weight and parameters['default'][1] is bias
    assert len(agent.actor_parameters) == 1 and agent.actor_parameters[0] is weight
    assert torch.equal(bias, parameters['bias'])
    assert not torch.equal(weight, parameters['weight'])

    w = agent.policy.get_weights()
    w_test = torch.tensor([-0.00764989, 1.23425770, 0.10249085])

    assert torch.allclose(w, w_test)
