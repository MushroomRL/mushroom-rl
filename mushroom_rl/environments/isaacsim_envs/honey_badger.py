import torch
from pathlib import Path

from mushroom_rl.environments.isaacsim_envs.quadruped import QuadrupedIsaac
from mushroom_rl.environments.isaacsim_envs.quadruped_randomizer import QuadrupedRandomizationParams
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import ObservationType


class HoneyBadgerIsaac(QuadrupedIsaac):
    """
    A learning environment for training the Honey Badger quadruped to walk.
    Honey Badger is a Robot from MAB Robotics: https://www.mabrobotics.pl/

    Args:
        num_envs (int): Number of parallel environments.
        horizon (int): The maximum horizon for the environment.
        domain_randomization (bool): Whether the domain randomization is enabled.
        camera_pos (tuple, None): The position of the camera looking at the scene.
        camera_target (tuple, None): The point the camera looking at the scene points to.
        reward_weights (dict, None): Overrides for the reward weights, on top of the ones this robot sets.
        **quadruped_params: Further parameters of :class:`QuadrupedIsaac`.

    """
    def __init__(self, num_envs, horizon, domain_randomization=True, camera_pos=None, camera_target=None,
                 reward_weights=None, **quadruped_params):
        usd_path, action_spec, default_joint_angles, default_joint_max_vel, trunk_body, foot_bodies, \
            sub_bodies, collision_groups = self._robot_config()

        reward_weights = dict() if reward_weights is None else reward_weights
        badger_reward_weights = dict(
            tracking_lin_vel=1.2, tracking_ang_vel=0.6, lin_vel_z=-1.6, ang_vel_xy=-0.07, torques=-1e-5,
            joint_acc=-2.5e-7, action_rate=-0.005, joint_pos_limits=-0.1, collision=-1.0, feet_air_time=0.,
            flat_orientation=-0.2, joint_vel_limits=-0.1, power_draw=-4e-6, similar_to_default=-0.085,
            stand_still_deviation=-1.4, feet_air_time_high=0.125, feet_air_time_low=0.75,
            feet_air_time_symmetry=-0.25, feet_clearance=0.085, feet_clearance_lateral=0.1, feet_slide=-0.04,
            feet_slide_low=-0.2, feet_z_velocity=-0.03, long_contact=-0.7, stand_still_short_contact=-0.5,
            height=-50.
        )

        observation_spec = [
            ("base_lin_vel", "", ObservationType.BODY_LIN_VEL, None),
            ("base_ang_vel", "", ObservationType.BODY_ANG_VEL, None),
            ("joint_pos", "", ObservationType.JOINT_POS, action_spec),
            ("joint_vel", "", ObservationType.JOINT_VEL, action_spec),
            ("base_pos", "", ObservationType.BODY_POS, None),
        ]
        additional_data_spec = [
            ("body_rot", "", ObservationType.BODY_ROT, None),
            ("body_vel", "", ObservationType.BODY_VEL, None),
        ]

        quadruped_params.setdefault("randomization_params", QuadrupedRandomizationParams(
            add_trunk_mass=(-1.5, 3.0), stay_at_default_percentage=0.3,
            torque_limit_factor=0.3, joint_velocity_factor=0.15, add_scaling_factor=(-0.03, 0.03),
            joint_damping_factor=0.5, joint_stiffness_factor=0.5, joint_armature_factor=0.5,
            motor_strength_factor=0.25
        ))

        super().__init__(usd_path, action_spec, default_joint_angles, trunk_body, foot_bodies, sub_bodies,
                         observation_spec, additional_data_spec, collision_groups, num_envs, horizon,
                         domain_randomization, camera_pos, camera_target,
                         default_joint_max_vel=default_joint_max_vel,
                         reward_weights=badger_reward_weights | reward_weights,
                         **quadruped_params)

    def is_absorbing(self, obs):
        forces = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)
        fallen = torch.any(torch.norm(forces, dim=-1) > 0., dim=-1)
        return fallen

    # observations ------------------------------------------------------------------------------------------------

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        base_pos_indices = self._observation_helper.obs_idx_map["base_pos"]
        obs[:, base_pos_indices[:2]] = 0

        return obs

    # reward function -----------------------------------------------------------------------------------------

    def _extra_reward_terms(self, next_obs):
        base_pos = self._observation_helper.get_from_obs(next_obs, "base_pos")
        base_pos_z = base_pos[:, 2]
        return self._reward_height(base_pos_z) * self._reward_weights["height"] * self.dt

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        forces = self._collision_helper.get_net_contact_forces("lower_body", dt=self._timestep)
        contact = torch.norm(forces, dim=-1) > 0.1
        return torch.sum(contact, dim=1)

    def _foot_contacts(self):
        return self._collision_helper.get_net_contact_forces("feet", dt=self._timestep)[:, :, 2] > 1.

    def _reward_height(self, base_z):
        # nominal_base_z = 0.316
        nominal_base_z = 0.31
        return torch.square(base_z - nominal_base_z)

    # utilities -----------------------------------------------------------------------------------------------

    @staticmethod
    def _robot_config():
        """
        Returns:
            The robot-specific configuration a :class:`HoneyBadgerIsaac` family member is built from, as the
            tuple ``(usd_path, action_spec, default_joint_angles, default_joint_max_vel, trunk_body,
            foot_bodies, sub_bodies, collision_groups)``. Overridden by subclasses simulating a different robot
            of the same family.

        """
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/honey_badger/honey_badger.usd")
        action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",
            "rl_j0", "rl_j1", "rl_j2",
            "rr_j0", "rr_j1", "rr_j2"
        ]
        device = TorchUtils.get_device()
        default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5
        ], device=device)
        default_joint_max_vel = torch.tensor([25.] * 12, device=device)
        trunk_body = "base_link"
        foot_bodies = ["/fl_foot", "/fr_foot", "/rl_foot", "/rr_foot"]
        sub_bodies = [
            "base_link",
            "fl_l0", "fr_l0", "rl_l0", "rr_l0",
            "fl_l1", "fr_l1", "rl_l1", "rr_l1",
            "fl_l2", "fr_l2", "rl_l2", "rr_l2",
            "fl_foot", "fr_foot", "rl_foot", "rr_foot"
        ]
        collision_groups = [
            ("feet", ["/fl_foot", "/fr_foot", "/rl_foot", "/rr_foot"]),
            ("body", ["/base_link", "/fl_l1", "/fr_l1", "/rl_l1", "/rr_l1"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]
        return usd_path, action_spec, default_joint_angles, default_joint_max_vel, trunk_body, foot_bodies, \
            sub_bodies, collision_groups
