import pytest

pytest.importorskip("mushroom_rl.environments.mujoco_envs")

from mushroom_rl.environments.mujoco_envs.reach import Reach
from mushroom_rl.environments.mujoco_envs.pick import Pick
from mushroom_rl.environments.mujoco_envs.push import Push
from mushroom_rl.environments.mujoco_envs.peg_insertion import PegInsertion
import numpy as np
import random
import torch


def test_reach():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    obs = []
    mdp = Reach()
    mdp.reset()
    action = np.zeros(7)

    for _ in range(20):
        observation, _, _, _ = mdp.step(action)
        assert len(observation) == len(mdp._mdp_info.observation_space.low)
        assert len(observation) == len(mdp._mdp_info.observation_space.high)
        obs.append(observation)

    obs_test = np.load("tests/environments/mujoco_envs/reach_data.npy")

    assert np.allclose(obs, obs_test)


def test_pick():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    obs = []
    mdp = Pick()
    mdp.reset()
    action = np.zeros(8)

    for _ in range(20):
        observation, _, _, _ = mdp.step(action)
        assert len(observation) == len(mdp._mdp_info.observation_space.low)
        assert len(observation) == len(mdp._mdp_info.observation_space.high)
        obs.append(observation)

    obs_test = np.load("tests/environments/mujoco_envs/pick_data.npy")

    assert np.allclose(obs, obs_test)


def test_push():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    obs = []
    mdp = Push()
    mdp.reset()
    action = np.zeros(7)

    for _ in range(20):
        observation, _, _, _ = mdp.step(action)
        assert len(observation) == len(mdp._mdp_info.observation_space.low)
        assert len(observation) == len(mdp._mdp_info.observation_space.high)
        obs.append(observation)

    obs_test = np.load("tests/environments/mujoco_envs/push_data.npy")

    assert np.allclose(obs, obs_test)


def test_peg_insertion():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    obs = []
    mdp = PegInsertion()
    mdp.reset()
    action = np.zeros(7)

    for _ in range(20):
        observation, _, _, _ = mdp.step(action)
        assert len(observation) == len(mdp._mdp_info.observation_space.low)
        assert len(observation) == len(mdp._mdp_info.observation_space.high)
        obs.append(observation)

    obs_test = np.load("tests/environments/mujoco_envs/peg_insertion_data.npy")

    assert np.allclose(obs, obs_test)


def test_push_reward():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    mdp = Push()
    mdp.reset()
    action = np.zeros(7)
    action[0] = 2.17
    action[1] = 2.17

    rewards = []
    contact_costs = []
    robot_forces = []
    gripper_forces = []

    for _ in range(100):
        observation, reward, _, info = mdp.step(action)
        assert np.ndim(reward) == 0
        rewards.append(reward)
        contact_costs.append(info["contact_cost"])
        robot_forces.append(mdp.obs_helper.get_from_obs(observation, "robot_contact_force").copy())
        gripper_forces.append(mdp.obs_helper.get_from_obs(observation, "gripper_contact_force").copy())

    robot_forces = np.array(robot_forces)
    gripper_forces = np.array(gripper_forces)
    dual_contact_steps = np.sum(np.any(robot_forces != 0, axis=1) & np.any(gripper_forces != 0, axis=1))

    assert robot_forces.shape == (100, 3)
    assert gripper_forces.shape == (100, 3)
    assert dual_contact_steps == 43
    assert np.sum(np.any(robot_forces != 0, axis=1)) == 52
    assert np.sum(np.any(gripper_forces != 0, axis=1)) == 67
    assert np.allclose(np.abs(robot_forces).max(), 368.51741001728493)
    assert np.allclose(np.abs(gripper_forces).max(), 173.14783792360254)
    assert np.allclose(np.sum(rewards), -119.28108591360848)
    assert np.allclose(np.min(contact_costs), -0.0006)
    assert np.allclose(np.sum(contact_costs), -0.035400174085850264)


def test_pick_reward():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    mdp = Pick()
    mdp.reset()
    action = np.zeros(8)
    action[0] = 2.17
    action[1] = 2.17

    rewards = []
    contact_costs = []
    robot_forces = []
    gripper_forces = []

    for _ in range(100):
        observation, reward, _, info = mdp.step(action)
        assert np.ndim(reward) == 0
        rewards.append(reward)
        contact_costs.append(info["contact_cost"])
        robot_forces.append(mdp.obs_helper.get_from_obs(observation, "robot_contact_force").copy())
        gripper_forces.append(mdp.obs_helper.get_from_obs(observation, "gripper_contact_force").copy())

    robot_forces = np.array(robot_forces)
    gripper_forces = np.array(gripper_forces)
    dual_contact_steps = np.sum(np.any(robot_forces != 0, axis=1) & np.any(gripper_forces != 0, axis=1))

    assert robot_forces.shape == (100, 3)
    assert gripper_forces.shape == (100, 3)
    assert dual_contact_steps == 41
    assert np.sum(np.any(robot_forces != 0, axis=1)) == 52
    assert np.sum(np.any(gripper_forces != 0, axis=1)) == 65
    assert np.allclose(np.abs(robot_forces).max(), 280.46010599025664)
    assert np.allclose(np.abs(gripper_forces).max(), 344.9527000232645)
    assert np.allclose(np.sum(rewards), 0.8374257001944749)
    assert np.allclose(np.min(contact_costs), -0.0006)
    assert np.allclose(np.sum(contact_costs), -0.035100000000000006)


def test_peg_insertion_reward():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    mdp = PegInsertion()
    mdp.reset()
    action = np.zeros(7)
    action[0] = 2.17
    action[1] = 2.17

    rewards = []
    robot_forces = []
    gripper_forces = []

    for _ in range(100):
        observation, reward, _, info = mdp.step(action)
        assert np.ndim(reward) == 0
        rewards.append(reward)
        robot_forces.append(mdp.obs_helper.get_from_obs(observation, "robot_collision_force").copy())
        gripper_forces.append(mdp.obs_helper.get_from_obs(observation, "gripper_collision_force").copy())

    robot_forces = np.array(robot_forces)
    gripper_forces = np.array(gripper_forces)
    dual_contact_steps = np.sum(np.any(robot_forces != 0, axis=1) & np.any(gripper_forces != 0, axis=1))

    assert robot_forces.shape == (100, 3)
    assert gripper_forces.shape == (100, 3)
    assert dual_contact_steps == 23
    assert np.sum(np.any(robot_forces != 0, axis=1)) == 49
    assert np.sum(np.any(gripper_forces != 0, axis=1)) == 41
    assert np.allclose(np.abs(robot_forces).max(), 489.6709347216338)
    assert np.allclose(np.abs(gripper_forces).max(), 214.6805301749962)
    assert np.allclose(np.sum(rewards), 4.085399761591603)


def test_peg_insertion_alignment():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    mdp = PegInsertion()
    observation = mdp.reset()[0]

    aligned = observation.copy()
    mdp.obs_helper.get_from_obs(aligned, "rel_goal_pos")[:] = [0.0, 0.0, 0.05]

    misaligned = observation.copy()
    mdp.obs_helper.get_from_obs(misaligned, "rel_goal_pos")[:] = [0.5, 0.5, 0.05]

    mdp._obs = misaligned

    assert mdp._is_aligned(aligned)
    assert np.allclose(mdp._get_insertion_reward(aligned), 8.068242641099854)

    mdp._obs = aligned

    assert not mdp._is_aligned(misaligned)
    assert np.allclose(mdp._get_insertion_reward(misaligned), 0.0)
