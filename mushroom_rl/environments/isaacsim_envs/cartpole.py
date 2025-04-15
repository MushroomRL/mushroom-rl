from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType
from mushroom_rl.core import ArrayBackend

import numpy as np
from pathlib import Path

class CartPole(IsaacSim):
    def __init__(self, num_envs, headless=True, backend="torch", device="cuda:0", camera_pos=(20, 0, 4), camera_target=(10, 0, 0)):
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
        collision_between_envs = False
        env_spacing = 2.5
        super().__init__(usd_path, action_spec, observation_spec, backend, device, collision_between_envs, num_envs, 
                         env_spacing, 0.99, 200, additional_data_spec=additional_data_spec, headless=headless,
                         camera_position=camera_pos, camera_target=camera_target)
        
        self.backend = ArrayBackend.get_array_backend(backend)
        
    def reward(self, obs, action, next_obs, absorbing):
        pole_joint_pos = self.observation_helper.get_from_obs(next_obs, "poleJointPos").squeeze()
        cart_joint_pos = self.observation_helper.get_from_obs(next_obs, "cartJointPos").squeeze()
        reward = 1.0 - self.backend.abs(cart_joint_pos)
        reward = self.backend.where(absorbing, -self.backend.ones_like(pole_joint_pos), reward)
        return reward

    def is_absorbing(self, obs):
        pole_joint_pos = self.observation_helper.get_from_obs(obs, "poleJointPos").squeeze()
        ones = self.backend.ones_like(pole_joint_pos, dtype=bool)
        zeros = self.backend.zeros_like(pole_joint_pos, dtype=bool)
        dropped = self.backend.where(self.backend.abs(pole_joint_pos) > np.pi / 2, ones, zeros)
        return dropped

    def setup(self, env_indices, obs):
        num_environments = len(env_indices)

        cart_joint_pos = 0.25 * (2.0 * self.backend.rand(num_environments, 1, device=self._device) - 1)
        pole_joint_pos = 0.05 * np.pi * (2.0 * self.backend.rand(num_environments, 1, device=self._device) - 1)

        cart_joint_vel = 0.25 * (2.0 * self.backend.rand(num_environments, 1, device=self._device) - 1)
        pole_joint_vel = 0.05 * np.pi * (2.0 * self.backend.rand(num_environments, 1, device=self._device) - 1)

        self._write_data("cartJointPos", cart_joint_pos, env_indices)
        self._write_data("poleJointPos", pole_joint_pos, env_indices)
        self._write_data("cartJointVel", cart_joint_vel, env_indices)
        self._write_data("poleJointVel", pole_joint_vel, env_indices)
    
    def _create_info_dictionary(self, obs):
        info = {}
        info["cartPosition"] = self._read_data("cartPos")
        info["polePosition"] = self._read_data("polePos")
        info["poleAngularVelocity"] = self._read_data("poleAngVel")
        return info