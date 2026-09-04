import pytest

pytest.importorskip("mushroom_rl.environments.mujoco_warp_envs")

import torch
import warp as wp

import numpy as np

from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.environments.mujoco_warp_envs import AntWarp, HalfCheetahWarp, HopperWarp, Walker2DWarp


def _rollout(mdp, num_envs, n_steps):
    mask = torch.ones(num_envs, dtype=torch.bool, device='cpu')
    action_dim = mdp.info.action_space.shape[0]

    obs, _ = mdp.reset_all(mask)
    for _ in range(n_steps):
        action = torch.rand(num_envs, action_dim, device='cpu')
        obs, reward, absorbing, _ = mdp.step_all(mask, action)

    return obs


def test_ant():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
        torch.manual_seed(1)
        mdp = AntWarp(num_envs=4, use_graph_capture=False)

        obs = _rollout(mdp, num_envs=4, n_steps=10)
        obs_test = np.load('tests/environments/mujoco_warp_envs/locomotion_ant_data.npy')

        assert np.allclose(obs.numpy(), obs_test)
    finally:
        TorchUtils.set_default_device('cpu')


def test_half_cheetah():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
        torch.manual_seed(1)
        mdp = HalfCheetahWarp(num_envs=4, use_graph_capture=False)

        obs = _rollout(mdp, num_envs=4, n_steps=10)
        obs_test = np.load('tests/environments/mujoco_warp_envs/locomotion_half_cheetah_data.npy')

        assert np.allclose(obs.numpy(), obs_test)
    finally:
        TorchUtils.set_default_device('cpu')


def test_hopper():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
        torch.manual_seed(1)
        mdp = HopperWarp(num_envs=4, use_graph_capture=False)

        obs = _rollout(mdp, num_envs=4, n_steps=10)
        obs_test = np.load('tests/environments/mujoco_warp_envs/locomotion_hopper_data.npy')

        assert np.allclose(obs.numpy(), obs_test)
    finally:
        TorchUtils.set_default_device('cpu')


def test_walker_2d():
    TorchUtils.set_default_device('cpu')
    wp.set_device('cpu')
    try:
        torch.manual_seed(1)
        mdp = Walker2DWarp(num_envs=4, use_graph_capture=False)

        obs = _rollout(mdp, num_envs=4, n_steps=10)
        obs_test = np.load('tests/environments/mujoco_warp_envs/locomotion_walker2d_data.npy')

        assert np.allclose(obs.numpy(), obs_test)
    finally:
        TorchUtils.set_default_device('cpu')
