from pathlib import Path

import numpy as np
from collections import deque

from mushroom_rl.environments.mujoco_envs.humanoid_gait.utils import convert_traj_quat_to_euler
from .trajectory import HumanoidTrajectory


class GoalRewardInterface:
    """
    Interface to specify a reward function for the ``HumanoidGait`` environment.

    """
    def __call__(self, state, action, next_state):
        """
        Compute the reward.

        Args:
            state (np.ndarray): last state;
            action (np.ndarray): applied action;
            next_state (np.ndarray): current state.

        Returs:
            The reward for the current transition.

        """
        raise NotImplementedError

    def get_observation_space(self):
        """
        Getter.

        Returns:
             The low and high arrays of the observation space

        """
        obs = self.get_observation()
        return -np.inf * np.ones(len(obs)), np.inf * np.ones(len(obs))

    def get_observation(self):
        """
        Getter.

        Returns:
             The current observation.

        """
        return np.array([])

    def is_absorbing(self, state):
        """
        Getter.

        Returns:
            Whether the current state is absorbing or not.

        """
        return False

    def update_state(self):
        """
        Update the state of the object after each transition.

        """
        pass

    def reset_state(self):
        """
        Reset the state of the object.

        """
        pass


class NoGoalReward(GoalRewardInterface):
    """
    Implement a reward function that is always 0.

    """
    def __call__(self, state, action, next_state):
        return 0


class MaxVelocityReward(GoalRewardInterface):
    """
    Implement a goal reward for achieving the maximum possible velocity.

    """
    def __init__(self, sim, traj_start=False, **kwargs):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            traj_start (bool, False): If model initial position should be set
                from a valid trajectory state. If False starts from the
                model.xml base position;
            **kwargs: additional parameters which can be passed to
                trajectory when using ``traj_start``. ``traj_path`` should be
                given to select a different trajectory. Rest of the arguments
                are not important.

        """
        self.traj_start = traj_start

        if traj_start:
            if "traj_path" not in kwargs:
                traj_path = Path(__file__).resolve().parent.parent.parent /\
                            "data" / "humanoid_gait" / "gait_trajectory.npz"
                kwargs["traj_path"] = traj_path.as_posix()
            self.trajectory = HumanoidTrajectory(sim, **kwargs)
            self.reset_state()

    def __call__(self, state, action, next_state):
        return next_state[13]

    def reset_state(self):
        if self.traj_start:
            self.trajectory.reset_trajectory()


class VelocityProfileReward(GoalRewardInterface):
    """
    Implement a goal reward for following a velocity profile.

    """
    def __init__(self, sim, profile_instance, traj_start=False, **kwargs):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            profile_instance (VelocityProfile): Velocity profile to
                follow. See RewardGoals.velocity_profile.py;
            traj_start (bool, False): If model initial position should be set
                from a valid trajectory state. If False starts from the
                model.xml base position;
            **kwargs: additional parameters which can be passed to
                trajectory when using ``traj_start``. ``traj_path`` should be
                given to select a diferent trajectory. Rest of the arguments
                are not important.

        """
        self.profile = profile_instance
        self.velocity_profile = deque(self.profile.values)

        self.traj_start = traj_start
        if traj_start:
            if "traj_path" not in kwargs:
                traj_path = Path(__file__).resolve().parent.parent.parent /\
                            "data" / "humanoid_gait" / "gait_trajectory.npz"
                kwargs["traj_path"] = traj_path.as_posix()
            self.trajectory = HumanoidTrajectory(sim, **kwargs)
            self.reset_state()

    def __call__(self, state, action, next_state):
        return np.exp(-np.linalg.norm(next_state[13:16] - self.velocity_profile[0]))

    def update_state(self):
        self.velocity_profile.rotate(1)

    def get_observation(self):
        return self.velocity_profile[0]

    def reset_state(self):
        self.velocity_profile = deque(self.profile.reset())

        if self.traj_start:
            substep_no = np.argmin(
                    np.linalg.norm(
                            self.trajectory.velocity_profile
                            - np.expand_dims(self.velocity_profile[0], 1),
                            axis=0))
            self.trajectory.reset_trajectory(substep_no=substep_no)

    def plot_velocity_profile_example(self, horizon=1000, n_trajectories=10):
        values = np.zeros((horizon * n_trajectories, 3))
        i = 0
        for t in range(n_trajectories):
            self.reset_state()
            for s in range(horizon):
                self.update_state()
                values[i, :] = self.get_observation()
                i += 1

        self.reset_state()
        import matplotlib.pyplot as plt
        plt.plot(values)
        for line_pos in [i*horizon for i in range(n_trajectories)]:
            plt.axvline(line_pos, c="red", alpha=0.3)
        plt.legend(["axis x", "axis y", "axis z"])
        plt.show()


class CompleteTrajectoryReward(GoalRewardInterface, HumanoidTrajectory):
    """
    Implements a goal reward for matching a kinematic trajectory.

    """
    def __init__(self, sim, control_dt=0.005, traj_path=None,
                 traj_dt=0.0025, traj_speed_mult=1.0,
                 use_error_terminate=False, **kwargs):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            control_dt (float, 0.005): frequency of the controller;
            traj_path (string, None): path with the trajectory for the
                model to follow. If None is passed, use default
                trajectory;
            traj_dt (float, 0.0025): time step of the trajectory file;
            traj_speed_mult (float, 1.0): factor to speed up or slowdown the
                trajectory velocity;
            use_error_terminate (bool, False): If episode should be terminated
                when the model deviates significantly from the reference
                 trajectory.

        """
        if traj_path is None:
            traj_path = Path(__file__).resolve().parent.parent.parent / "data" / "humanoid_gait" / "gait_trajectory.npz"
            traj_path = traj_path.as_posix()

        super(CompleteTrajectoryReward, self).__init__(sim, traj_path, traj_dt,
                                                       control_dt,
                                                       traj_speed_mult)
        self.error_terminate = use_error_terminate

        self.error_threshold = 0.20
        self.terminate_trajectory_flag = False

        self.euler_traj = convert_traj_quat_to_euler(self.subtraj)

        self.traj_data_range = np.clip(2 * np.std(self.euler_traj, axis=1), 0.15, np.inf)

        self.joint_importance = np.where(self.traj_data_range < 0.15, 2 * self.traj_data_range, 1.0)
        self.joint_importance[2 :14] *= 1.0
        self.joint_importance[17:28] *= 0.1
        self.joint_importance[28:34] *= 5.0
        self.joint_importance = np.r_[self.joint_importance[2:14],
                                      self.joint_importance[17:28],
                                      self.joint_importance[28:34]]

        self.traj_data_range = np.concatenate([self.traj_data_range[2:14],
                                               self.traj_data_range[17:34]])

    def __call__(self, state, action, next_state):
        traj_reward_vec = self._calculate_each_comp_reward(state, action,
                                                           next_state)

        norm_traj_reward = np.sum(traj_reward_vec * self.joint_importance) / np.sum(
                self.joint_importance)

        if self.error_terminate and norm_traj_reward < (1 - self.error_threshold):
            self.terminate_trajectory_flag = True

        if self.error_terminate:
            norm_traj_reward = 1 + (norm_traj_reward - 1) / self.error_threshold
        return norm_traj_reward

    def _calculate_each_comp_reward(self, state, action, next_state):
        euler_state = convert_traj_quat_to_euler(next_state, offset=2)

        foot_vec = np.append(
            (self.sim.data.body_xpos[1] - self.sim.data.body_xpos[4]),
            (self.sim.data.body_xpos[1] - self.sim.data.body_xpos[7])
        )

        current_state = np.concatenate([euler_state[0:12],
                                        euler_state[15:26], foot_vec])

        current_target = np.concatenate(
            [self.euler_traj[2:14, self.subtraj_step_no],
            self.euler_traj[17:34, self.subtraj_step_no]]
        )

        current_error_standard = (np.subtract(current_state, current_target) /
                                  self.traj_data_range)

        traj_reward_vec = np.exp(-np.square(current_error_standard))
        return traj_reward_vec

    def update_state(self):
        self.subtraj_step_no += 1
        if self.subtraj_step_no >= self.traj_length:
            self.get_next_sub_trajectory()

    def get_observation(self):
        return self.velocity_profile[:, self.subtraj_step_no]

    def is_absorbing(self, state):
        return self.terminate_trajectory_flag

    def reset_state(self):
        self.reset_trajectory()
        self.terminate_trajectory_flag = False
