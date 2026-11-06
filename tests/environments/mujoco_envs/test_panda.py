import random

import numpy as np
import pytest
import torch

from helper.envs import constant_rollout

pytest.importorskip("mujoco")

from mushroom_rl.environments.mujoco_envs.manipulation import Reach, Pick, Push, PegInsertion  # noqa: E402


def seed_all():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)


@pytest.mark.parametrize('env, n_actions, name', [(Reach, 7, 'reach'), (Pick, 8, 'pick'), (Push, 7, 'push'),
                                                  (PegInsertion, 7, 'peg_insertion')])
def test_rollout(env, n_actions, name):
    seed_all()
    observations = constant_rollout(env(), np.zeros(n_actions))

    assert np.allclose(observations, np.load(f'tests/environments/mujoco_envs/data/test_rollout_{name}.npy'))


def test_additional_data():
    seed_all()

    mdp = PegInsertion()
    mdp.reset()

    sizes = {key: len(mdp._read_data(key)) for key in mdp.additional_data}

    assert sizes == {'peg_pos': 3, 'peg_rot': 4, 'goal_pos': 3, 'goal_rot': 4, 'joint1_pos': 1, 'joint2_pos': 1,
                     'joint3_pos': 1, 'joint4_pos': 1, 'joint5_pos': 1, 'joint6_pos': 1, 'joint7_pos': 1,
                     'finger_joint1_pos': 1, 'finger_joint2_pos': 1, 'gripper_pos': 3, 'gripper_rot': 9}


def test_rewards():
    seed_all()

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
    assert dual_contact_steps == 41
    assert np.sum(np.any(robot_forces != 0, axis=1)) == 50
    assert np.sum(np.any(gripper_forces != 0, axis=1)) == 64
    assert np.allclose(np.abs(robot_forces).max(), 455.9975975309887)
    assert np.allclose(np.abs(gripper_forces).max(), 337.3089184619118)
    assert np.allclose(np.sum(rewards), -119.24321498332925)
    assert np.allclose(np.min(contact_costs), -0.0006)
    assert np.allclose(np.sum(contact_costs), -0.034047468372260595)

    seed_all()

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
    assert np.sum(np.any(robot_forces != 0, axis=1)) == 51
    assert np.sum(np.any(gripper_forces != 0, axis=1)) == 65
    assert np.allclose(np.abs(robot_forces).max(), 275.6706714331601)
    assert np.allclose(np.abs(gripper_forces).max(), 144.35769423737185)
    assert np.allclose(np.sum(rewards), 0.8314528651984001)
    assert np.allclose(np.min(contact_costs), -0.0006)
    assert np.allclose(np.sum(contact_costs), -0.03443473778441179)

    seed_all()

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
    assert dual_contact_steps == 22
    assert np.sum(np.any(robot_forces != 0, axis=1)) == 47
    assert np.sum(np.any(gripper_forces != 0, axis=1)) == 33
    assert np.allclose(np.abs(robot_forces).max(), 612.5917962548951)
    assert np.allclose(np.abs(gripper_forces).max(), 186.04840014236282)
    assert np.allclose(np.sum(rewards), 4.159753405350865)


def test_alignment():
    seed_all()

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
