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
from mushroom_rl.environments.isaacsim_envs import A1Isaac, Go2Isaac, HoneyBadgerIsaac, SilverBadgerIsaac
from mushroom_rl.environments.isaacsim_envs.quadruped_randomizer import QuadrupedRandomizationParams


def run_env(mdp, num_joints):
    n_envs = mdp.number
    mask = torch.ones(n_envs, dtype=torch.bool, device="cuda:0")

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

    mdp.stop()

    return obs.cpu().numpy()


def test_randomization_params():
    params = QuadrupedRandomizationParams()

    assert params["torque_limit_factor"] == 0.
    assert params["joint_damping"] == (0.0, 0.3)
    assert "p_gain_scale" in params
    assert "nominal_p_gain" not in params
    assert "torque_limt_factor" not in params

    overridden = QuadrupedRandomizationParams(torque_limit_factor=0.5, joint_damping=(0.1, 0.2))

    assert overridden["torque_limit_factor"] == 0.5
    assert overridden["joint_damping"] == (0.1, 0.2)
    assert overridden["p_gain_scale"] == (0.85, 1.15)
    assert params["torque_limit_factor"] == 0.

    with pytest.raises(ValueError):
        QuadrupedRandomizationParams(torque_limt_factor=0.5)

    with pytest.raises(KeyError):
        params["torque_limt_factor"]


def test_cartpole():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = CartPoleIsaac(n_envs)

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

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)
    obs_test = np.load('tests/environments/isaacsim_envs/a1_data.npy')

    assert np.allclose(obs, obs_test)


def test_go2():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = Go2Isaac(n_envs, 1000)

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)
    obs_test = np.load('tests/environments/isaacsim_envs/go2_data.npy')

    assert np.allclose(obs, obs_test)


def test_honey_badger():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = HoneyBadgerIsaac(n_envs, 1000)

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)
    obs_test = np.load('tests/environments/isaacsim_envs/honey_badger_data.npy')

    assert np.allclose(obs, obs_test)


def test_silver_badger():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = SilverBadgerIsaac(n_envs, 1000)

    assert mdp.number == n_envs

    obs = run_env(mdp, 13)
    obs_test = np.load('tests/environments/isaacsim_envs/silver_badger_data.npy')

    assert np.allclose(obs, obs_test)


def test_honey_badger_no_domain_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = HoneyBadgerIsaac(n_envs, 1000, domain_randomization=False)

    assert mdp.number == n_envs

    obs = run_env(mdp, 12)

    assert np.all(np.isfinite(obs))


def test_silver_badger_no_domain_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    n_envs = 2
    mdp = SilverBadgerIsaac(n_envs, 1000, domain_randomization=False)

    assert mdp.number == n_envs

    obs = run_env(mdp, 13)

    assert np.all(np.isfinite(obs))


def test_set_domain_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    names = ('trunk_mass', 'trunk_inertia', 'trunk_com', 'torque_limit', 'max_joint_vel', 'joint_damping',
             'joint_stiffness', 'joint_armature', 'joint_friction')
    envs = torch.arange(2, device='cuda:0')
    mask = torch.ones(2, dtype=torch.bool, device='cuda:0')

    mdp = Go2Isaac(2, 1000, domain_randomization=False)
    mdp.set_domain_randomization(False)
    nominal = {name: mdp._observation_helper.read_data(name, envs).clone() for name in names}
    nominal_gains = (mdp._randomizer.p_gain.clone(), mdp._randomizer.d_gain.clone())
    mdp.reset_all(mask)
    nominal_friction = mdp._robots._physics_articulation_view.get_material_properties().numpy().copy()
    mdp.stop()

    mdp = Go2Isaac(2, 1000)
    mdp.reset_all(mask)
    mdp.reset_all(mask)
    randomized_mass = mdp._observation_helper.read_data('trunk_mass', envs).clone()

    mdp.set_domain_randomization(False)
    mdp.reset_all(mask)

    for name in names:
        assert torch.equal(mdp._observation_helper.read_data(name, envs), nominal[name]), name
    assert np.array_equal(mdp._robots._physics_articulation_view.get_material_properties().numpy(), nominal_friction)
    assert torch.equal(mdp._randomizer.p_gain, nominal_gains[0])
    assert torch.equal(mdp._randomizer.d_gain, nominal_gains[1])
    assert torch.all(mdp._randomizer.motor_strength == 1.) and torch.all(mdp._randomizer.position_offset == 0.)
    assert torch.all(mdp._randomizer.delay_steps == 0)

    mdp.set_domain_randomization(True)
    mass = mdp._observation_helper.read_data('trunk_mass', envs)

    assert not torch.equal(mass, nominal['trunk_mass']) and not torch.equal(mass, randomized_mass)
    mdp.stop()


def test_joint_friction_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    envs = torch.arange(2, device='cuda:0')
    mask = torch.ones(2, dtype=torch.bool, device='cuda:0')

    mdp = Go2Isaac(2, 1000, randomization_params=QuadrupedRandomizationParams(joint_friction_factor=0.1),
                   observed_randomization=('joint_friction', ))
    nominal = mdp._randomizer.default_parameters['joint_friction']
    mdp.reset_all(mask)
    friction = mdp._observation_helper.read_data('joint_friction', envs)
    ratio = friction / nominal

    assert torch.equal(friction, mdp._randomizer.seen_parameters['joint_friction'])
    assert torch.allclose(ratio[..., 1], ratio[..., 0]) and torch.allclose(ratio[..., 2], ratio[..., 0])
    assert torch.allclose(friction[0, 0], torch.tensor([0.21405327, 0.21405327, 0.10702664], device='cuda:0'))
    assert len(mdp._observation_helper.obs_idx_map['joint_friction']) == 36

    mdp.set_domain_randomization(False)
    mdp.reset_all(mask)

    assert torch.equal(mdp._observation_helper.read_data('joint_friction', envs), nominal.expand(2, -1, -1))
    mdp.stop()


def test_termination_penalty():
    np.random.seed(1)
    torch.manual_seed(1)

    mask = torch.ones(2, dtype=torch.bool, device='cuda:0')
    absorbing = torch.tensor([True, False], device='cuda:0')

    mdp = Go2Isaac(2, 1000, clamp_reward=True, reward_weights=dict(termination=-1000.))
    obs, _ = mdp.reset_all(mask)
    next_obs, _, _, _ = mdp.step_all(mask, torch.zeros(2, 12, device='cuda:0'))
    reward = mdp.reward(obs, torch.zeros(2, 12, device='cuda:0'), next_obs, absorbing)

    assert torch.allclose(mdp._extra_info_rewards['termination'], torch.tensor([-20., 0.], device='cuda:0'))
    assert torch.allclose(reward, torch.tensor([-20., 0.], device='cuda:0'))
    mdp.stop()


def test_observed_randomization():
    np.random.seed(1)
    torch.manual_seed(1)

    mdp = A1Isaac(2, 1000)
    for name in ("p_gain", "torque_limit", "mass", "joint_damping"):
        assert name not in mdp._observation_helper.obs_idx_map
    mdp.stop()

    mdp = Go2Isaac(2, 1000)
    assert len(mdp._observation_helper.obs_idx_map["actual_delay"]) == 1
    assert len(mdp._observation_helper.obs_idx_map["joint_calib_offset"]) == 12
    for name in ("p_gain", "torque_limit", "mass", "joint_damping"):
        assert name not in mdp._observation_helper.obs_idx_map
    mdp.stop()

    mdp = A1Isaac(2, 1000, observed_randomization=("p_gain", "mass"))
    assert len(mdp._observation_helper.obs_idx_map["p_gain"]) == 12
    assert len(mdp._observation_helper.obs_idx_map["mass"]) == 1
    mdp.stop()

    with pytest.raises(ValueError):
        A1Isaac(2, 1000, observed_randomization=("p_gian", ))


def test_observation_indices():
    np.random.seed(1)
    torch.manual_seed(1)

    mdp = A1Isaac(2, 1000)

    assert torch.equal(mdp.observation_indices('base_lin_vel', 'base_ang_vel'),
                       torch.tensor([0, 1, 2, 3, 4, 5], device='cuda:0'))
    assert torch.equal(mdp.observation_indices('joint_pos'),
                       torch.tensor([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], device='cuda:0'))
    assert torch.equal(mdp.observation_indices('actions'),
                       torch.tensor([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], device='cuda:0'))

    with pytest.raises(ValueError):
        mdp.observation_indices('base_lin_vel', 'base_pos')

    mdp.stop()
