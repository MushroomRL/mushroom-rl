import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent, Core
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import PPO_BPTT, RudinPPO
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import GaussianTorchPolicy, RecurrentGaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork, RecurrentActorNetwork, RecurrentCriticNetwork


def learn(alg, policy, alg_params):
    mdp = InvertedPendulum(horizon=50)
    agent = alg(mdp.info, policy, **alg_params)
    core = Core(agent, mdp)
    core.learn(n_steps=50, n_steps_per_fit=50)
    return agent


def make_bptt_setup(use_prev_action=False):
    n_features, n_hidden = 4, 4
    dim_env_state, dim_action = 2, 1  # InvertedPendulum

    policy = RecurrentGaussianTorchPolicy(
        network=RecurrentActorNetwork,
        input_shape=(dim_env_state,),
        output_shape=(dim_action,),
        policy_state_shape=(n_hidden,),
        n_features=n_features,
        n_hidden_features=n_hidden,
        rnn_type='gru',
        action_history_shape=(dim_action,) if use_prev_action else None,
    )

    alg_params = dict(
        actor_optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
        critic_params=dict(
            network=RecurrentCriticNetwork,
            optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
            loss=F.mse_loss,
            input_shape=(dim_env_state,),
            output_shape=(1,),
            policy_state_shape=(n_hidden,),
            n_features=n_features,
            n_hidden_features=n_hidden,
            rnn_type='gru',
            action_history_shape=(dim_action,) if use_prev_action else None,
        ),
        n_epochs_policy=2, batch_size=64, eps_ppo=.2, lam=.95,
        dim_env_state=dim_env_state,
        action_history_length=1 if use_prev_action else 0,
    )

    return policy, alg_params


def make_rudin_setup():
    policy = GaussianTorchPolicy(
        ActorNetwork,
        (2,),
        (1,),
        std_0=1., n_features=None, n_layers=0,
    )

    alg_params = dict(
        actor_optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
        critic_params=dict(
            network=ActorNetwork,
            optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
            loss=F.mse_loss,
            n_features=None, n_layers=0,
            input_shape=(2,), output_shape=(1,),
        ),
        n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95,
        clip_grad_norm=1., schedule='adaptive', desired_kl=0.01,
    )

    return policy, alg_params


def test_PPO_BPTT():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    policy, alg_params = make_bptt_setup()

    w = learn(PPO_BPTT, policy, alg_params).policy.get_weights().numpy()
    w_test = np.load('tests/algorithms/test_ppo_bptt.npy')

    assert np.allclose(w, w_test, atol=1e-4), f'max discrepancy: {np.max(np.abs(w - w_test))}, w[:5]={w[:5]}'


def test_PPO_BPTT_prev_action():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    policy, alg_params = make_bptt_setup(use_prev_action=True)

    w = learn(PPO_BPTT, policy, alg_params).policy.get_weights().numpy()
    w_test = np.load('tests/algorithms/test_ppo_bptt_prev_action.npy')

    assert np.allclose(w, w_test, atol=6e-4), f'max discrepancy: {np.max(np.abs(w - w_test))}, w[:5]={w[:5]}'


def test_PPO_BPTT_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    policy, alg_params = make_bptt_setup()

    agent_save = learn(PPO_BPTT, policy, alg_params)
    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att in vars(agent_save):
        tu.assert_eq(getattr(agent_save, att), getattr(agent_load, att))


def test_RudinPPO():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    policy, alg_params = make_rudin_setup()

    w = learn(RudinPPO, policy, alg_params).policy.get_weights()
    w_test = torch.tensor([0.6614, -1.3338, -0.1395, -0.0024])

    assert torch.allclose(w, w_test, atol=1e-4)


def test_RudinPPO_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    policy, alg_params = make_rudin_setup()

    agent_save = learn(RudinPPO, policy, alg_params)
    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att in vars(agent_save):
        tu.assert_eq(getattr(agent_save, att), getattr(agent_load, att))
