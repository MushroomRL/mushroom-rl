import torch
from pathlib import Path

from mushroom_rl.environments.isaacsim_envs.quadruped import QuadrupedIsaac
from mushroom_rl.environments.isaacsim_envs.quadruped_randomizer import QuadrupedRandomizationParams
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import ObservationType


class A1Isaac(QuadrupedIsaac):
    """
    A learning environment for training the A1 quadruped to walk.

    Resembles environment implemented by Rudin et al. for
    "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"

    Args:
        num_envs (int): Number of parallel environments.
        horizon (int): The maximum horizon for the environment.
        domain_randomization (bool): Whether the domain randomization is enabled.
        camera_position (tuple, None): The position of the camera looking at the scene.
        camera_target (tuple, None): The point the camera looking at the scene points to.
        **quadruped_params: Further parameters of :class:`QuadrupedIsaac`, e.g. the reward weights.

    """
    def __init__(self, num_envs, horizon, domain_randomization=True, camera_position=None, camera_target=None,
                 **quadruped_params):
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/a1/a1.usd")
        device = TorchUtils.get_device()

        action_spec = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]
        default_joint_angles = torch.tensor([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1., -1.5,
            -0.1, 1., -1.5
        ], device=device)
        trunk_body = "base_link"
        foot_bodies = ["/FL_foot", "/FR_foot", "/RL_foot", "/RR_foot"]
        sub_bodies = [
            "base_link",
            "FL_hip", "FR_hip", "RL_hip", "RR_hip",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
            "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_foot", "FR_foot", "RL_foot", "RR_foot"
        ]
        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),

            ("joint_pos", "", ObservationType.JOINT_POS, action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, action_spec)
        ]
        additional_data_spec = [
            ("body_rot", "", ObservationType.BODY_ROT, None),
            ("body_vel", "", ObservationType.BODY_VEL, None)
        ]

        collision_groups = [
            ("feet", ["/FL_foot", "/FR_foot", "/RL_foot", "/RR_foot"]),
            ("body", ["/base_link"]),
            ("lower_body", ["/FL_thigh", "/FR_thigh", "/RL_thigh", "/RR_thigh",
                            "/FL_calf", "/FR_calf", "/RL_calf", "/RR_calf"])
        ]

        quadruped_params.setdefault("randomization_params",
                                    QuadrupedRandomizationParams(add_trunk_mass=(-1.3, 2.6)))

        super().__init__(usd_path, action_spec, default_joint_angles, trunk_body, foot_bodies, sub_bodies,
                         observation_spec, additional_data_spec, collision_groups, num_envs, horizon,
                         domain_randomization, camera_position, camera_target, **quadruped_params)

    def is_absorbing(self, obs):
        trunk_forces = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)[:, 0]
        fallen = torch.norm(trunk_forces, dim=-1) > 1.
        return fallen

    # reward function -----------------------------------------------------------------------------------------

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        forces = self._collision_helper.get_net_contact_forces("lower_body", dt=self._timestep)
        contact = torch.norm(forces, dim=-1) > 0.1
        return torch.sum(contact, dim=1)

    def _foot_contacts(self):
        return self._collision_helper.get_net_contact_forces("feet", dt=self._timestep)[:, :, 2] > 1.
