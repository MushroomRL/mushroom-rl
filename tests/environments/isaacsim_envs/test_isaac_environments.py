import pytest

pytest.importorskip("isaacsim")

import numpy as np
import torch

from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import IsaacLauncher

# Isaac Sim has to be running before its environments can be imported
IsaacLauncher.launch(headless=True)
TorchUtils.set_default_device("cuda:0")

from mushroom_rl.environments.isaacsim_envs import CartPoleIsaac
from mushroom_rl.environments.isaacsim_envs import A1Isaac, HoneyBadgerIsaac, SilverBadgerIsaac
from mushroom_rl.environments.isaacsim_envs.quadruped_randomizer import QuadrupedRandomizationParams


def run_env(mdp, num_joints):
    n_envs = mdp.number
    mask = torch.ones(n_envs, device="cuda:0")

    obs, _ = mdp.reset_all(mask)
    assert isinstance(obs, torch.Tensor) and obs.is_cuda
    assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
    assert obs.shape == (n_envs, len(mdp.info.observation_space.high))

    for i in range(20):
        if i < 10:
            action = torch.tensor([[0.] * num_joints] * n_envs, device="cuda:0")
        else:
            action = torch.tensor([[1.] * num_joints] * n_envs, device="cuda:0")

        obs, reward, absorbing, _ = mdp.step_all(mask, action)

        assert isinstance(obs, torch.Tensor) and obs.is_cuda
        assert isinstance(reward, torch.Tensor) and reward.is_cuda
        assert isinstance(absorbing, torch.Tensor) and absorbing.is_cuda

        assert obs.shape == (n_envs, len(mdp.info.observation_space.low))
        assert obs.shape == (n_envs, len(mdp.info.observation_space.high))
        assert reward.shape == (n_envs, )
        assert absorbing.shape == (n_envs, )

    mdp.stop(soft=False)

    return obs.cpu().numpy()


def test_randomization_params():
    params = QuadrupedRandomizationParams()

    assert params["torque_limit_factor"] == 0.3
    assert params["joint_damping"] == (0.0, 0.3)
    assert "add_p_gain" in params
    assert "nominal_p_gain" not in params
    assert "torque_limt_factor" not in params

    overridden = QuadrupedRandomizationParams(torque_limit_factor=0.5, joint_damping=(0.1, 0.2))

    assert overridden["torque_limit_factor"] == 0.5
    assert overridden["joint_damping"] == (0.1, 0.2)
    assert overridden["add_p_gain"] == (-3.0, 3.0)
    assert params["torque_limit_factor"] == 0.3

    with pytest.raises(ValueError):
        QuadrupedRandomizationParams(torque_limt_factor=0.5)

    with pytest.raises(KeyError):
        params["torque_limt_factor"]


def test_cartpole():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = CartPoleIsaac(n_envs)
    mdp.seed(1)

    assert mdp.number == n_envs
    assert isinstance(mdp.info.observation_space.low, torch.Tensor)
    assert isinstance(mdp.info.observation_space.high, torch.Tensor)
    assert isinstance(mdp.info.action_space.low, torch.Tensor)
    assert isinstance(mdp.info.action_space.high, torch.Tensor)
    assert mdp.info.observation_space.low.is_cuda
    assert mdp.info.observation_space.high.is_cuda
    assert mdp.info.action_space.low.is_cuda
    assert mdp.info.action_space.high.is_cuda

    obs = run_env(mdp, 1)
    obs_test = np.load('tests/environments/isaacsim_envs/cartpole_data.npy')

    assert np.allclose(obs, obs_test)


def test_a1():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = A1Isaac(n_envs, 1000)
    mdp.seed(1)

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)
    obs_test = np.load('tests/environments/isaacsim_envs/a1_data.npy')

    assert np.allclose(obs, obs_test)


def test_honey_badger():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = HoneyBadgerIsaac(n_envs, 1000)
    mdp.seed(1)

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)
    obs_test = np.load('tests/environments/isaacsim_envs/honey_badger_data.npy')

    assert np.allclose(obs, obs_test)


def test_silver_badger():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = SilverBadgerIsaac(n_envs, 1000)
    mdp.seed(1)

    assert mdp.number == n_envs

    obs = run_env(mdp, 13)
    obs_test = np.load('tests/environments/isaacsim_envs/silver_badger_data.npy')

    assert np.allclose(obs, obs_test)


def test_honey_badger_no_domain_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = HoneyBadgerIsaac(n_envs, 1000, domain_randomization=False)
    mdp.seed(1)

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)

    assert np.all(np.isfinite(obs))


def test_silver_badger_no_domain_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = SilverBadgerIsaac(n_envs, 1000, domain_randomization=False)
    mdp.seed(1)

    assert mdp.number == n_envs

    obs = run_env(mdp, 13)

    assert np.all(np.isfinite(obs))
