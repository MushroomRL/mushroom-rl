import pytest

pytest.importorskip("mushroom_rl.environments.mujoco_envs")

import numpy as np

from mushroom_rl.environments import Ant, HalfCheetah, Hopper, Walker2D

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"


def test_ant():
    np.random.seed(1)
    mdp = Ant()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.rand()])
    ns_test = np.load('tests/environments/mujoco_envs/locomotion_ant_data.npy')

    assert np.allclose(ns, ns_test)


def test_half_cheetah():
    np.random.seed(1)
    mdp = HalfCheetah()
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, *_ = mdp.step([np.random.rand()])
        states.append(ns)
        snapshots.append(np.copy(ns))
    for state, snapshot in zip(states, snapshots):
        assert np.allclose(state, snapshot)
    ns_test = np.load('tests/environments/mujoco_envs/locomotion_half_cheetah_data.npy')

    assert np.allclose(ns, ns_test)


def test_hopper():
    np.random.seed(1)
    mdp = Hopper()
    mdp.reset()
    for i in range(10):
        ns, *_ = mdp.step([np.random.rand()])
    ns_test = np.load('tests/environments/mujoco_envs/locomotion_hopper_data.npy')

    assert np.allclose(ns, ns_test)


def test_walker_2d():
    np.random.seed(1)
    mdp = Walker2D()
    mdp.reset()
    for i in range(10):
        ns, *_ = mdp.step([np.random.rand()])
    ns_test = np.load('tests/environments/mujoco_envs/locomotion_walker2d_data.npy')

    assert np.allclose(ns, ns_test)
