from mushroom_rl.environments.isaacsim_env import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType
from mushroom_rl.utils import TorchUtils

import math
import torch
from pathlib import Path


class CartPoleIsaac(IsaacSim):
    """
    Cart-pole balancing task simulated with Isaac Sim.

    The agent is rewarded for keeping the cart close to the origin, and the episode ends when the pole falls past
    the horizontal.

    """
    def __init__(self, num_envs, camera_pos=(20, 0, 4), camera_target=(10, 0, 0)):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/cartpole/cartpole.usd")
        action_spec = ["rail_cart_joint"]
        observation_spec = [
            ("poleJointPos", "", ObservationType.JOINT_POS, "cart_pole_joint"),
            ("poleJointVel", "", ObservationType.JOINT_VEL, "cart_pole_joint"),
            ("cartJointPos", "", ObservationType.JOINT_POS, "rail_cart_joint"),
            ("cartJointVel", "", ObservationType.JOINT_VEL, "rail_cart_joint")
        ]
        additional_data_spec = [
            ("cartPos", "/cart", ObservationType.BODY_POS, None),
            ("polePos", "/pole", ObservationType.BODY_POS, None),
            ("poleAngVel", "/pole", ObservationType.BODY_ANG_VEL, None)
        ]
        scene_params = dict(env_spacing=2.5)
        viewer_params = dict(camera_position=camera_pos, camera_target=camera_target)
        super().__init__(usd_path, action_spec, observation_spec, num_envs, 0.99, 200,
                         additional_data_spec=additional_data_spec, scene_params=scene_params,
                         viewer_params=viewer_params)

    def reward(self, obs, action, next_obs, absorbing):
        pole_joint_pos = self._observation_helper.get_from_obs(next_obs, "poleJointPos").squeeze()
        cart_joint_pos = self._observation_helper.get_from_obs(next_obs, "cartJointPos").squeeze()
        reward = 1.0 - torch.abs(cart_joint_pos)
        reward = torch.where(absorbing, -torch.ones_like(pole_joint_pos), reward)
        return reward

    def is_absorbing(self, obs):
        pole_joint_pos = self._observation_helper.get_from_obs(obs, "poleJointPos").squeeze()
        ones = torch.ones_like(pole_joint_pos, dtype=bool)
        zeros = torch.zeros_like(pole_joint_pos, dtype=bool)
        dropped = torch.where(torch.abs(pole_joint_pos) > math.pi / 2, ones, zeros)
        return dropped

    def setup(self, env_indices, obs):
        num_environments = len(env_indices)

        cart_joint_pos = 0.25 * (2.0 * torch.rand(num_environments, 1, device=TorchUtils.get_device()) - 1)
        pole_joint_pos = 0.05 * math.pi * (2.0 * torch.rand(num_environments, 1, device=TorchUtils.get_device()) - 1)

        cart_joint_vel = 0.25 * (2.0 * torch.rand(num_environments, 1, device=TorchUtils.get_device()) - 1)
        pole_joint_vel = 0.05 * math.pi * (2.0 * torch.rand(num_environments, 1, device=TorchUtils.get_device()) - 1)

        self._observation_helper.write_data("cartJointPos", cart_joint_pos, env_indices)
        self._observation_helper.write_data("poleJointPos", pole_joint_pos, env_indices)
        self._observation_helper.write_data("cartJointVel", cart_joint_vel, env_indices)
        self._observation_helper.write_data("poleJointVel", pole_joint_vel, env_indices)

    def _create_info_dictionary(self, obs):
        info = {"cartPosition": self._observation_helper.read_data("cartPos"),
                "polePosition": self._observation_helper.read_data("polePos"),
                "poleAngularVelocity": self._observation_helper.read_data("poleAngVel")}
        return info
