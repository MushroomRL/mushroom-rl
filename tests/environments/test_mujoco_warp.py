import pytest

pytest.importorskip("mushroom_rl.environments.mujoco_warp_envs")

import torch
import warp as wp

from mushroom_rl.core import Environment
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.environments.mujoco_warp_envs import AntWarp, HopperWarp


def test_registration():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
        mdp = Environment.make('HopperWarp', num_envs=2, use_graph_capture=False)

        assert isinstance(mdp, HopperWarp)
        assert mdp.number == 2
    finally:
        TorchUtils.set_default_device('cpu')


def test_reset_all_and_step_all_shapes():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
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
    finally:
        TorchUtils.set_default_device('cpu')


def test_partial_reset_leaves_other_envs_untouched():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
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
    finally:
        TorchUtils.set_default_device('cpu')


def test_seed_reproducibility():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
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
    finally:
        TorchUtils.set_default_device('cpu')


def test_graph_capture_matches_eager_stepping():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
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
    finally:
        TorchUtils.set_default_device('cpu')


def test_collision_detection():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
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
    finally:
        TorchUtils.set_default_device('cpu')
