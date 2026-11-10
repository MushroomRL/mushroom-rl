import pytest
import torch

from mushroom_rl.core import Environment
from mushroom_rl.environments import MuJoCoWarp
from mushroom_rl.environments.assets import asset_path

pytest.importorskip("mujoco_warp")

import warp as wp  # noqa: E402
from mushroom_rl.environments.mujoco_warp_envs import AntWarp, HopperWarp  # noqa: E402
from mushroom_rl.utils.mujoco import ObservationType  # noqa: E402


def test_interface():
    wp.set_device('cpu')
    mdp = Environment.make('HopperWarp', num_envs=2, use_graph_capture=False)

    assert isinstance(mdp, HopperWarp)
    assert mdp.number == 2

    torch.manual_seed(1)
    num_envs = 4
    mdp = HopperWarp(num_envs=num_envs, use_graph_capture=False)
    obs_dim = mdp.info.observation_space.shape[0]
    action_dim = mdp.info.action_space.shape[0]

    env_mask = torch.ones(num_envs, dtype=torch.bool, device='cpu')
    obs, info = mdp.reset_all(env_mask)

    assert obs.shape == (num_envs, obs_dim)
    assert obs.device.type == 'cpu'
    assert isinstance(info, dict)

    action = torch.rand(num_envs, action_dim, device='cpu')
    next_obs, reward, absorbing, info = mdp.step_all(env_mask, action)

    assert next_obs.shape == (num_envs, obs_dim)
    assert reward.shape == (num_envs,)
    assert absorbing.shape == (num_envs,)
    assert absorbing.dtype == torch.bool
    assert isinstance(info, dict)

    torch.manual_seed(4)
    mdp = HopperWarp(num_envs=1, use_graph_capture=False, reset_noise_scale=0.0)
    mask = torch.ones(1, dtype=torch.bool, device='cpu')
    mdp.reset_all(mask)
    single = mdp.step_all(mask, torch.full((3,), .5))[0]
    mdp.reset_all(mask)
    obs = torch.ones(1, 2)

    assert torch.equal(single, mdp.step_all(mask, torch.full((1, 3), .5))[0])
    assert mdp.get_all_observation_keys() == ['z_pos', 'y_pos', 'thigh_pos', 'leg_pos', 'foot_pos', 'x_vel', 'z_vel',
                                              'y_vel', 'thigh_vel', 'leg_vel', 'foot_vel']
    assert MuJoCoWarp._create_observation(mdp, obs) is obs
    assert MuJoCoWarp._create_info_dictionary(mdp, obs) == {}

    with pytest.raises(NotImplementedError):
        MuJoCoWarp.reward(mdp, None, None, None, None)
    with pytest.raises(NotImplementedError):
        MuJoCoWarp.is_absorbing(mdp, None)

    mdp.seed(3)
    first = torch.rand(1)
    mdp.seed(3)

    assert torch.equal(torch.rand(1), first)


def test_partial_reset():
    wp.set_device('cpu')
    torch.manual_seed(2)
    num_envs = 3
    mdp = HopperWarp(num_envs=num_envs, use_graph_capture=False)
    action_dim = mdp.info.action_space.shape[0]

    all_mask = torch.ones(num_envs, dtype=torch.bool, device='cpu')
    obs, _ = mdp.reset_all(all_mask)
    action = torch.rand(num_envs, action_dim, device='cpu')
    for _ in range(3):
        obs, _, _, _ = mdp.step_all(all_mask, action)

    reset_mask = torch.tensor([True, False, False], device='cpu')
    new_obs, _ = mdp.reset_all(reset_mask)

    assert not torch.equal(new_obs[0], obs[0])
    assert torch.equal(new_obs[1:], obs[1:])

    torch.manual_seed(3)
    num_envs = 4
    mdp = HopperWarp(num_envs=num_envs, use_graph_capture=False, reset_noise_scale=0.0)
    obs_dim = mdp.info.observation_space.shape[0]

    mdp.reset_all(torch.ones(num_envs, dtype=torch.bool, device='cpu'))

    reset_mask = torch.tensor([False, True, False, True], device='cpu')
    state = torch.full((num_envs, obs_dim), 99.0, device='cpu')
    state[1] = torch.arange(1, obs_dim + 1, device='cpu') * 0.01
    state[3] = -torch.arange(1, obs_dim + 1, device='cpu') * 0.01

    obs, _ = mdp.reset_all(reset_mask, state)

    assert torch.equal(obs[1], state[1])
    assert torch.equal(obs[3], state[3])
    assert (obs[0] != 99.0).all()
    assert (obs[2] != 99.0).all()

    torch.manual_seed(7)
    num_envs = 4
    mdp = AntWarp(num_envs=num_envs, use_graph_capture=False, reset_noise_scale=0.0)
    obs_dim = mdp.info.observation_space.shape[0]

    assert mdp.obs_helper.build_omit_idx['root_pose'] == [0, 1]

    mdp.reset_all(torch.ones(num_envs, dtype=torch.bool, device='cpu'))

    reset_mask = torch.tensor([False, True, False, True], device='cpu')
    state = torch.full((num_envs, obs_dim), 99.0, device='cpu')
    state[1] = torch.arange(1, obs_dim + 1, device='cpu') * 0.01
    state[3] = -torch.arange(1, obs_dim + 1, device='cpu') * 0.01

    obs, _ = mdp.reset_all(reset_mask, state)

    assert torch.equal(obs[1], state[1])
    assert torch.equal(obs[3], state[3])
    assert (obs[0] != 99.0).all()
    assert (obs[2] != 99.0).all()


def test_seed():
    wp.set_device('cpu')
    num_envs = 3

    torch.manual_seed(4)
    mdp_a = AntWarp(num_envs=num_envs, use_graph_capture=False)
    mask = torch.ones(num_envs, dtype=torch.bool, device='cpu')
    obs_a, _ = mdp_a.reset_all(mask)
    action = torch.rand(num_envs, mdp_a.info.action_space.shape[0], device='cpu')
    for _ in range(5):
        obs_a, _, _, _ = mdp_a.step_all(mask, action)

    torch.manual_seed(4)
    mdp_b = AntWarp(num_envs=num_envs, use_graph_capture=False)
    obs_b, _ = mdp_b.reset_all(mask)
    for _ in range(5):
        obs_b, _, _, _ = mdp_b.step_all(mask, action)

    assert torch.equal(obs_a, obs_b)


def test_graph_capture():
    wp.set_device('cpu')
    num_envs = 3

    torch.manual_seed(5)
    mdp_eager = AntWarp(num_envs=num_envs, use_graph_capture=False)
    mask = torch.ones(num_envs, dtype=torch.bool, device='cpu')
    obs_eager, _ = mdp_eager.reset_all(mask)
    action = torch.rand(num_envs, mdp_eager.info.action_space.shape[0], device='cpu')
    for _ in range(5):
        obs_eager, _, _, _ = mdp_eager.step_all(mask, action)

    torch.manual_seed(5)
    mdp_graph = AntWarp(num_envs=num_envs, use_graph_capture=True)
    obs_graph, _ = mdp_graph.reset_all(mask)
    for _ in range(5):
        obs_graph, _, _, _ = mdp_graph.step_all(mask, action)

    assert torch.equal(obs_eager, obs_graph)


def test_collisions():
    wp.set_device('cpu')
    torch.manual_seed(6)
    num_envs = 2
    mdp = AntWarp(num_envs=num_envs, use_graph_capture=False)
    mask = torch.ones(num_envs, dtype=torch.bool, device='cpu')
    mdp.reset_all(mask)

    collided = mdp._check_collision('torso', 'floor')
    force = mdp._get_collision_force('torso', 'floor')

    assert collided.shape == (num_envs,)
    assert collided.dtype == torch.bool
    assert not collided.any()
    assert force.shape == (num_envs, 6)
    assert torch.equal(force, torch.zeros_like(force))

    mdp = AntWarp(num_envs=num_envs, use_graph_capture=False, reset_noise_scale=0.)
    state, _ = mdp.reset_all(mask)
    state[0, 0] = .1
    mdp.reset_all(mask, state)

    assert torch.equal(mdp._check_collision('torso', 'floor'), torch.tensor([True, False]))
    assert torch.equal(mdp._check_collision('floor', 'torso'), torch.tensor([True, False]))

    force = mdp._get_collision_force('torso', 'floor')
    expected = torch.zeros(num_envs, 6, dtype=torch.float32)
    expected[0, :3] = torch.tensor([300.4520263671875, -7.62939453125e-06, 1.52587890625e-05])
    assert force.dtype == torch.float32
    assert torch.equal(force, expected)
    assert torch.equal(mdp._get_collision_force('floor', 'torso'), expected)

    mdp.step_all(mask, torch.zeros(num_envs, 8, device='cpu'))

    assert not mdp._check_collision('torso', 'floor').any()

    mdp = MuJoCoWarp(asset_path('ant', 'model.xml'), ['hip_1'],
                     [('root_pose', 'root', ObservationType.JOINT_POS)], .99, 100, num_envs,
                     collision_groups=[('body', ['torso_geom', 'left_ankle_geom', 'right_ankle_geom',
                                                 'third_ankle_geom', 'fourth_ankle_geom']), ('floor', ['floor'])])
    state, _ = mdp.reset_all(mask)
    state[0, 2] = .2
    state[0, 3:5] = torch.tensor([.98, .2])
    mdp.reset_all(mask, state)
    mdp._mj_warp.forward(mdp._model_wp, mdp._data_wp)
    expected = torch.zeros(num_envs, 6)
    expected[0, :2] = 36.0725555419921875

    assert torch.equal(mdp._check_collision('body', 'floor'), torch.tensor([True, False]))
    assert torch.equal(mdp._get_collision_force('body', 'floor'), expected)

    mdp = MuJoCoWarp('tests/environments/data/flex_cloth.xml', [],
                     [('box_pose', 'box_joint', ObservationType.JOINT_POS)], .99, 100, num_envs,
                     collision_groups=[('box', ['box_geom']), ('floor', ['floor'])])
    mdp.reset_all(mask)
    mdp._mj_warp.forward(mdp._model_wp, mdp._data_wp)

    assert wp.to_torch(mdp._data_wp.nacon)[0] == 18
    assert not mdp._check_collision('box', 'floor').any()
    assert not mdp._get_collision_force('box', 'floor').any()

    mdp = MuJoCoWarp('tests/environments/data/distance_sensor.xml', [],
                     [('ball_pose', 'ball_joint', ObservationType.JOINT_POS)], .99, 100, num_envs,
                     collision_groups=[('ball', ['ball']), ('floor', ['floor'])])
    mdp.reset_all(mask)
    mdp._mj_warp.forward(mdp._model_wp, mdp._data_wp)

    assert wp.to_torch(mdp._data_wp.nacon)[0] == 2
    assert not mdp._check_collision('ball', 'floor').any()
    assert not mdp._get_collision_force('ball', 'floor').any()
