import time

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from scipy import signal, interpolate


class Trajectory(object):
    """
    Builds a general trajectory from a numpy bin file(.npy), and automatically
    synchronizes the trajectory timestep to the desired control timestep while
    also allowing to change it's speed by the desired amount. When using
    periodic trajectories it is also possible to pass split points which signal
    the points where the trajectory repeats, and provides an utility to select
    the desired cycle.

    """
    def __init__(self, traj_path, traj_dt=0.01, control_dt=0.01,
                 traj_speed_mult=1.0):
        """
        Constructor.

        Args:
            traj_path (string): path with the trajectory for the
                model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a
                'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations);
            traj_dt (float, 0.01): time step of the trajectory file;
            control_dt (float, 0.01): model control frequency (used to
                synchronize trajectory with the control step);
            traj_speed_mult (float, 1.0): factor to speed up or slowdown the
                trajectory velocity.

        """
        trajectory_files = np.load(traj_path)
        self.trajectory = trajectory_files["trajectory_data"]

        if "split_points" in trajectory_files.files:
            self.split_points = trajectory_files["split_points"]
        else:
            self.split_points = np.array([0, self.trajectory.shape[1]])

        self.n_repeating_steps = len(self.split_points) - 1

        self.traj_dt = traj_dt
        self.control_dt = control_dt
        self.traj_speed_multiplier = traj_speed_mult

        if self.traj_dt != control_dt or traj_speed_mult != 1.0:
            new_traj_sampling_factor = (1 / traj_speed_mult) * (
                    self.traj_dt / control_dt)

            self.trajectory = self._interpolate_trajectory(
                self.trajectory, factor=new_traj_sampling_factor
            )

            self.split_points = np.round(
                self.split_points * new_traj_sampling_factor).astype(np.int32)

    def _interpolate_trajectory(self, traj, factor):
        x = np.arange(traj.shape[1])
        x_new = np.linspace(0, traj.shape[1] - 1, round(traj.shape[1] * factor),
                            endpoint=True)
        new_traj = interpolate.interp1d(x, traj, kind="cubic", axis=1)(x_new)
        return new_traj

    def _get_traj_gait_sub_steps(self, initial_walking_step,
                                 number_of_walking_steps=1):
        start_sim_step = self.split_points[initial_walking_step]
        end_sim_step = self.split_points[
            initial_walking_step + number_of_walking_steps
        ]

        sub_traj = self.trajectory[:, start_sim_step:end_sim_step].copy()
        initial_x_pos = self.trajectory[0][start_sim_step]
        sub_traj[0, :] -= initial_x_pos
        return sub_traj


class HumanoidTrajectory(Trajectory):
    """
    Loads a trajectory to be used by the humanoid environment. The trajectory
    file should be structured as:
    trajectory[0:15] -> model's qpos;
    trajectory[15:29] -> model's qvel;
    trajectory[29:34] -> model's foot vector position
    trajectory[34:36] -> model's ground force reaction over z

    """
    def __init__(self, sim, traj_path, traj_dt=0.0025,
                 control_dt=0.005, traj_speed_mult=1.0,
                 velocity_smooth_window=1001):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            traj_path (string): path with the trajectory for the
                model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a
                'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations);
            traj_dt (float, 0.0025): time step of the trajectory file;
            control_dt (float, 0.005): Model control frequency(used to
                synchronize trajectory with the control step)
            traj_speed_mult (float, 1.0): factor to speed up or slowdown the
                trajectory velocity;
            velocity_smooth_window (int, 1001): size of window used to average
                the torso velocity. It is used in order to get the average
                travelling velocity(as walking velocity from humanoids
                are sinusoidal).

        """
        super().__init__(traj_path, traj_dt, control_dt, traj_speed_mult)

        self.sim = sim
        self.trajectory[15:29] *= traj_speed_mult

        self.complete_velocity_profile = self._smooth_vel_profile(
                self.trajectory[15:18],  window_size=velocity_smooth_window)

        self.subtraj_step_no = 0
        self.x_dist = 0

        self.subtraj = self.trajectory.copy()
        self.velocity_profile = self.complete_velocity_profile.copy()
        self.reset_trajectory()

    @property
    def traj_length(self):
        return self.subtraj.shape[1]

    def _get_traj_gait_sub_steps(self, initial_walking_step,
                                 number_of_walking_steps=1):
        start_sim_step = self.split_points[initial_walking_step]
        end_sim_step = self.split_points[
            initial_walking_step + number_of_walking_steps
        ]

        sub_traj = self.trajectory[:, start_sim_step:end_sim_step].copy()
        initial_x_pos = self.trajectory[0][start_sim_step]
        sub_traj[0, :] -= initial_x_pos

        sub_vel_profile = self.complete_velocity_profile[
                          :, start_sim_step:end_sim_step].copy()

        return sub_traj, sub_vel_profile

    def _smooth_vel_profile(self, vel, use_simple_mean=False, window_size=1001,
                            polyorder=2):
        if use_simple_mean:
            filtered = np.tile(np.mean(vel, axis=1),
                               reps=(self.trajectory.shape[1], 1)).T
        else:
            filtered = signal.savgol_filter(vel, window_length=window_size,
                                            polyorder=polyorder, axis=1)
        return filtered

    def reset_trajectory(self, substep_no=None):
        """
        Resets the trajectory and the model. The trajectory can be forced
        to start on the 'substep_no' if desired, else it starts at
        a random one.

        Args:
            substep_no (int, None): starting point of the trajectory.
                If None, the trajectory starts from a random point.
        """
        self.x_dist = 0
        if substep_no is None:
            self.subtraj_step_no = int(np.random.rand() * (
                    self.traj_length * 0.45))
        else:
            self.subtraj_step_no = substep_no

        self.subtraj = self.trajectory.copy()
        self.subtraj[0, :] -= self.subtraj[0, self.subtraj_step_no]

        self.sim.data.qpos[0:15] = self.subtraj[0:15, self.subtraj_step_no]
        self.sim.data.qvel[0:14] = self.subtraj[15:29, self.subtraj_step_no]

    def get_next_sub_trajectory(self):
        """
        Get the next trajectory once the current one reaches it's end.

        """
        self.x_dist += self.subtraj[0][-1]
        self.reset_trajectory()

    def play_trajectory_demo(self, freq=200):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every step

        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = True
        self.reset_trajectory()
        while True:
            if self.subtraj_step_no >= self.traj_length:
                self.get_next_sub_trajectory()

            self.sim.data.qpos[0:15] = np.r_[
                self.x_dist + self.subtraj[0, self.subtraj_step_no],
                self.subtraj[1:15, self.subtraj_step_no]
            ]
            self.sim.data.qvel[0:14] = self.subtraj[15:29, self.subtraj_step_no]
            self.sim.forward()

            self.subtraj_step_no += 1
            time.sleep(1 / freq)
            viewer.render()

    def _plot_joint_trajectories(self, n_points=2000):
        """
        Plots the joint trajectories(qpos / qvel) in case the user wishes
            to consult them.

        """
        fig, ax = plt.subplots(2, 8, figsize=(15 * 8, 15))
        fig.suptitle("Complete Trajectories Sample", size=25)

        for j in range(8):
            ax[0, j].plot(self.subtraj[7 + j, 0:n_points])
            ax[0, j].legend(["Joint {} pos".format(j)])

            ax[1, j].plot(self.subtraj[7 + j + 14, 0:n_points])
            ax[1, j].plot(np.diff(
                self.subtraj[7 + j, 0:n_points]) / self.control_dt)
            ax[1, j].legend(["Joint {} vel".format(j), "derivate of pos"])
        plt.show()
