from pathlib import Path

import torch

from mushroom_rl.environments.isaacsim_envs.honey_badger import HoneyBadgerIsaac
from mushroom_rl.utils import TorchUtils


class SilverBadgerIsaac(HoneyBadgerIsaac):
    """
    A learning environment for training the Silver Badger quadroped to walk.
    Silver Badger is a Robot from MAB Robotics: https://www.mabrobotics.pl/

    Same robot family as :class:`HoneyBadgerIsaac`, only differing in the robot-specific configuration
    ``_robot_config`` returns: it adds the ``rear`` body and its extra ``sp_j0`` joint.

    """
    @staticmethod
    def _robot_config():
        usd_path = str(Path(__file__).resolve().parent / "robots_usds/silver_badger/silver_badger.usd")
        action_spec = [
            "fl_j0", "fl_j1", "fl_j2",
            "fr_j0", "fr_j1", "fr_j2",
            "rl_j0", "rl_j1", "rl_j2",
            "rr_j0", "rr_j1", "rr_j2",
            "sp_j0"
        ]
        device = TorchUtils.get_device()
        default_joint_angles = torch.tensor([
            0.1, -0.8, 1.5,
            -0.1, 0.8, -1.5,
            0.1, -1., 1.5,
            -0.1, 1., -1.5,
            0
        ], device=device)
        default_joint_max_vel = torch.tensor([25.] * 13, device=device)
        sub_bodies = [
            "body", "rear",
            "fl_l0", "fr_l0", "rl_l0", "rr_l0",
            "fl_l1", "fr_l1", "rl_l1", "rr_l1",
            "fl_l2", "fr_l2", "rl_l2", "rr_l2",
            "fl_foot", "fr_foot", "rl_foot", "rr_foot"
        ]
        collision_groups = [
            ("feet", ["/fl_foot", "/fr_foot", "/rl_foot", "/rr_foot"]),
            ("body", ["/body", "/rear", "/fl_l1", "/fr_l1", "/rl_l1", "/rr_l1"]),
            ("lower_body", ["/fl_l2", "/fr_l2", "/rl_l2", "/rr_l2"])
        ]
        return usd_path, action_spec, default_joint_angles, default_joint_max_vel, sub_bodies, collision_groups
