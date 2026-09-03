import torch
from pathlib import Path

from mushroom_rl.environments.isaacsim_envs.quadruped import QuadrupedIsaac
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import ObservationType


class Go2Isaac(QuadrupedIsaac):
    """
    A learning environment for training the Unitree Go2 quadruped to walk.

    The geometry of the robot comes from the model Unitree ships for MuJoCo, and its inertial properties and
    actuator limits from the urdf Unitree publishes, which describes the motor rotors the MuJoCo model leaves
    out and keeps the feet apart from the calves carrying them.

    On top of what every quadruped observes, this one observes the position of its trunk, of which only the
    height means anything: the reward term on the height of the trunk reads it, and a critic can be given it.
    A policy meant to be deployed should not, since the real robot cannot measure it, any more than it can
    measure the linear velocity of its trunk -- see the training example for how to hide both from it.

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
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/go2/go2.usd")
        device = TorchUtils.get_device()

        legs = ("FL", "FR", "RL", "RR")
        thigh_paths = {leg: f"/Geometry/{leg}_thigh" for leg in legs}
        calf_paths = {leg: f"/Geometry/{leg}_calf" for leg in legs}

        action_spec = [f"{leg}_{joint}_joint" for leg in legs for joint in ("hip", "thigh", "calf")]
        default_joint_angles = torch.tensor([
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1., -1.5,
            -0.1, 1., -1.5
        ], device=device)
        trunk_body = "base_link"
        foot_bodies = [f"/Geometry/{leg}_foot" for leg in legs]
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
            ("joint_vel", "", ObservationType.JOINT_VEL, action_spec),

            ("base_pos", "", ObservationType.BODY_POS, None)
        ]
        additional_data_spec = [
            ("body_rot", "", ObservationType.BODY_ROT, None),
            ("body_vel", "", ObservationType.BODY_VEL, None)
        ]

        collision_groups = [
            ("feet", list(foot_bodies)),
            ("body", ["/Geometry/base_link"]),
            ("lower_body", [thigh_paths[leg] for leg in legs] + [calf_paths[leg] for leg in legs])
        ]

        quadruped_params.setdefault("observed_randomization", ("actual_delay", "joint_calib_offset"))

        super().__init__(usd_path, action_spec, default_joint_angles, trunk_body, foot_bodies, sub_bodies,
                         observation_spec, additional_data_spec, collision_groups, num_envs, horizon,
                         domain_randomization, camera_position, camera_target, **quadruped_params)

    def is_absorbing(self, obs):
        trunk_forces = self._collision_helper.get_net_contact_forces("body", dt=self._timestep)[:, 0]
        fallen = torch.norm(trunk_forces, dim=-1) > 1.
        return fallen

    # construction-time hooks -------------------------------------------------------------------------------------

    def _domain_randomization_obs_bounds(self):
        bounds = super()._domain_randomization_obs_bounds()
        params = self._randomization_params
        position_offset = params["position_offset"]

        bounds["actual_delay"] = (1, 0., float(params["max_delay_steps"]))
        bounds["joint_calib_offset"] = (len(self._action_spec), -position_offset, position_offset)

        return bounds

    def _domain_randomization_obs_value(self, name):
        if name == "actual_delay":
            return self._randomizer.delay_steps.unsqueeze(1).float()

        if name == "joint_calib_offset":
            return self._randomizer.position_offset

        return super()._domain_randomization_obs_value(name)

    # observations ------------------------------------------------------------------------------------------------

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        # only the height of the trunk means anything: where in the world the robot walks does not
        base_pos_indices = self._observation_helper.obs_idx_map["base_pos"]
        obs[:, base_pos_indices[:2]] = 0

        return obs

    # reward function -----------------------------------------------------------------------------------------

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        forces = self._collision_helper.get_net_contact_forces("lower_body", dt=self._timestep)
        contact = torch.norm(forces, dim=-1) > 0.1
        return torch.sum(contact, dim=1)

    def _foot_contacts(self):
        return self._collision_helper.get_net_contact_forces("feet", dt=self._timestep)[:, :, 2] > 1.
