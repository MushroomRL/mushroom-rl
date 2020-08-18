import mujoco_py
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.running_stats import *

from ._external_simulation import NoExternalSimulation, MuscleSimulation
from .reward_goals import CompleteTrajectoryReward, VelocityProfileReward, \
     MaxVelocityReward, NoGoalReward, HumanoidTrajectory
from mushroom_rl.environments.mujoco_envs.humanoid_gait.utils import quat_to_euler


class HumanoidGait(MuJoCo):
    """
    Mujoco simulation of a Humanoid Model, based on:
    "A deep reinforcement learning based approach towards generating human
    walking behavior with a neuromuscular model".
    Anand, A., Zhao, G., Roth, H., and Seyfarth, A. (2019).

    """
    def __init__(self, gamma=0.99, horizon=2000, n_intermediate_steps=10,
                 use_muscles=True, goal_reward=None, goal_reward_params=None,
                 obs_avg_window=1, act_avg_window=1):
        """
        Constructor.

        Args:
            gamma (float, 0.99): discount factor for the environment;
            horizon (int, 2000): horizon for the environment;
            n_intermediate_steps (int, 10): number of steps to apply the same
                action to the environment and wait for the next observation;
            use_muscles (bool): if external muscle simulation should be used
                for actions. If not apply torques directly to the joints;
            goal_reward (string, None): type of trajectory used for training
                Options available:
                    'trajectory'         - Use trajectory in assets/GaitTrajectory.npz
                                           as reference;
                    'com_vel_trajectory' - Use only velocity trajectory of COM in
                                           assets/GaitTrajectory.npz as reference;
                    'vel_profile'        - Velocity goal for the center of mass of the
                                           model to follow. The goal is given by a
                                           VelocityProfile instance (or subclass).
                                           And should be included in the
                                           ``goal_reward_params``;
                    'max_vel'            - Tries to achieve the maximum possible
                                           velocity;
                    None                 - Follows no goal(just tries to survive);
            goal_reward_params (dict, None): params needed for creation goal
                reward;
            obs_avg_window (int, 1): size of window used to average
                observations;
            act_avg_window (int, 1): size of window used to average actions.

        """
        self.use_muscles = use_muscles
        self.goal_reward = goal_reward
        self.act_avg_window = act_avg_window
        self.obs_avg_window = obs_avg_window

        model_path = Path(__file__).resolve().parent.parent / "data" / "humanoid_gait" / "human7segment.xml"

        action_spec = ["right_hip_frontal", "right_hip_sagittal",
                       "right_knee", "right_ankle", "left_hip_frontal",
                       "left_hip_sagittal", "left_knee", "left_ankle",
                       ]

        observation_spec = [("root", ObservationType.JOINT_POS),
                            ("right_hip_frontal", ObservationType.JOINT_POS),
                            ("right_hip_sagittal", ObservationType.JOINT_POS),
                            ("right_knee", ObservationType.JOINT_POS),
                            ("right_ankle", ObservationType.JOINT_POS),
                            ("left_hip_frontal", ObservationType.JOINT_POS),
                            ("left_hip_sagittal", ObservationType.JOINT_POS),
                            ("left_knee", ObservationType.JOINT_POS),
                            ("left_ankle", ObservationType.JOINT_POS),

                            ("root", ObservationType.JOINT_VEL),
                            ("right_hip_frontal", ObservationType.JOINT_VEL),
                            ("right_hip_sagittal", ObservationType.JOINT_VEL),
                            ("right_knee", ObservationType.JOINT_VEL),
                            ("right_ankle", ObservationType.JOINT_VEL),
                            ("left_hip_frontal", ObservationType.JOINT_VEL),
                            ("left_hip_sagittal", ObservationType.JOINT_VEL),
                            ("left_knee", ObservationType.JOINT_VEL),
                            ("left_ankle", ObservationType.JOINT_VEL),
                            ]

        collision_groups = [("floor", ["floor"]),
                            ("left_foot", ["left_foot"]),
                            ("right_foot", ["right_foot"])
                            ]

        super().__init__(model_path.as_posix(), action_spec, observation_spec, gamma=gamma,
                         horizon=horizon, n_substeps=1,
                         n_intermediate_steps=n_intermediate_steps,
                         collision_groups=collision_groups)

        if use_muscles:
            self.external_actuator = MuscleSimulation(self._sim)
            self.info.action_space = spaces.Box(
                *self.external_actuator.get_action_space())
        else:
            self.external_actuator = NoExternalSimulation()

        low, high = self.info.action_space.low.copy(),\
                    self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

        if goal_reward_params is None:
            goal_reward_params = dict()

        if goal_reward == "trajectory" or goal_reward == "com_vel_trajectory":
            control_dt = self._sim.model.opt.timestep * self._n_intermediate_steps
            self.goal_reward = CompleteTrajectoryReward(self._sim, control_dt,
                                                        **goal_reward_params)
        elif goal_reward == "vel_profile":
            self.goal_reward = VelocityProfileReward(self._sim, **goal_reward_params)
        elif goal_reward == "max_vel":
            self.goal_reward = MaxVelocityReward(self._sim, **goal_reward_params)
        elif goal_reward is None:
            self.goal_reward = NoGoalReward()
        else:
            raise NotImplementedError("The specified goal reward has not been"
                                      "implemented: ", goal_reward)

        if goal_reward == "trajectory":
            self.reward_weights = dict(live_reward=0.10, goal_reward=0.40,
                                       traj_vel_reward=0.50,
                                       move_cost=0.10, fall_cost=0.00)
        elif goal_reward == "com_vel_trajectory":
            self.reward_weights = dict(live_reward=0.00, goal_reward=0.00,
                                       traj_vel_reward=1.00,
                                       move_cost=0.00, fall_cost=0.00)
        else:
            self.reward_weights = dict(live_reward=0.10, goal_reward=0.90,
                                       traj_vel_reward=0.00,
                                       move_cost=0.10, fall_cost=0.00)

        self.info.observation_space = spaces.Box(*self._get_observation_space())

        self.mean_grf = RunningAveragedWindow(shape=(6,),
                                              window_size=n_intermediate_steps)
        self.mean_vel = RunningExpWeightedAverage(shape=(3,), alpha=0.005)
        self.mean_obs = RunningAveragedWindow(
            shape=self.info.observation_space.shape,
            window_size=obs_avg_window
        )
        self.mean_act = RunningAveragedWindow(
            shape=self.info.action_space.shape, window_size=act_avg_window)

    def step(self, action):
        action = ((action.copy() * self.norm_act_delta) + self.norm_act_mean)

        state, reward, absorbing, info = super().step(action)

        self.mean_obs.update_stats(state)
        self.mean_vel.update_stats(self._sim.data.qvel[0:3])

        avg_obs = self.mean_obs.mean
        avg_obs[13:16] = self.mean_vel.mean
        return avg_obs, reward, absorbing, info

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)
            self._viewer._render_every_frame = True
        self._viewer.render()

    def _setup(self):
        self.goal_reward.reset_state()
        start_obs = self._reset_model(qpos_noise=0.0, qvel_noise=0.0)
        start_vel = (
            self._sim.data.qvel[0:3] if (self.goal_reward is None or isinstance(
                self.goal_reward, MaxVelocityReward)
                                         ) else self.goal_reward.get_observation())

        self.mean_vel.reset(start_vel)
        self.mean_obs.reset(start_obs)
        self.mean_act.reset()
        self.external_actuator.reset()

    def _reward(self, state, action, next_state):
        live_reward = 1.0

        goal_reward = self.goal_reward(state, action, next_state)

        traj_vel_reward = 0.0
        if isinstance(self.goal_reward, HumanoidTrajectory):
            traj_vel_reward = np.exp(-20.0 * np.square(
                next_state[13] - next_state[33]))

        move_cost = self.external_actuator.cost(
            state, action / self.norm_act_delta, next_state)

        fall_cost = 0.0
        if self._has_fallen(next_state):
            fall_cost = 1.0

        total_reward = self.reward_weights["live_reward"] * live_reward \
            + self.reward_weights["goal_reward"] * goal_reward \
            + self.reward_weights["traj_vel_reward"] * traj_vel_reward \
            - self.reward_weights["move_cost"] * move_cost \
            - self.reward_weights["fall_cost"] * fall_cost

        return total_reward

    def _is_absorbing(self, state):
        return (self._has_fallen(state)
                or self.goal_reward.is_absorbing(state)
                or self.external_actuator.is_absorbing(state)
                )

    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        grf_low, grf_high = (-np.ones((6,)) * np.inf,
                             np.ones((6,)) * np.inf)

        r_low, r_high = self.goal_reward.get_observation_space()

        a_low, a_high = self.external_actuator.get_observation_space()

        return (np.concatenate([sim_low, grf_low, r_low, a_low]),
                np.concatenate([sim_high, grf_high, r_high, a_high]))

    def _reset_model(self, qpos_noise=0.0, qvel_noise=0.0):
        self._set_state(self._sim.data.qpos + np.random.uniform(
            low=-qpos_noise, high=qpos_noise, size=self._sim.model.nq),
                        self._sim.data.qvel + np.random.uniform(low=-qvel_noise,
                                                                high=qvel_noise,
                                                                size=self._sim.model.nv)
                        )

        return self._create_observation()

    def _set_state(self, qpos, qvel):
        old_state = self._sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self._sim.set_state(new_state)
        self._sim.forward()

    @staticmethod
    def _has_fallen(state):
        torso_euler = quat_to_euler(state[1:5])
        return ((state[0] < 0.90) or (state[0] > 1.20)
                or abs(torso_euler[0]) > np.pi / 12
                or (torso_euler[1] < -np.pi / 12) or (torso_euler[1] > np.pi / 8)
                or (torso_euler[2] < -np.pi / 4) or (torso_euler[2] > np.pi / 4)
                )

    def _create_observation(self):
        """
        Creates full vector of observations:

        obs[0:13] -> qpos(from mujoco obs)
        obs[0] -> torso z pos
        obs[1:5] -> torso quaternion orientation
        obs[5:13] -> leg joints angle

        obs[13:27] -> qvel(from mujoco obs)
        obs[13:16] -> torso linear velocity
        obs[16:19] -> torso angular velocity
        obs[19:27] -> leg joints angular velocity

        obs[27:30] ->  ground force
        obs[27:30] -> ground force on right foot(xyz)
        obs[30:33] -> ground force on left foot(xyz)

        obs[33:33+(len(goal_observation)] -> observations related
                                             to the goal

        obs[last_obs_id - len(ext_actuator_obs): last_obs_id]
                -> observations related to the external actuator

        """
        obs = np.concatenate([super(HumanoidGait, self)._create_observation()[2:],
                              self.mean_grf.mean / 1000.,
                              self.goal_reward.get_observation(),
                              self.external_actuator.get_observation()
                              ]).flatten()
        return obs

    def _preprocess_action(self, action):
        action = self.external_actuator.preprocess_action(action)
        self.mean_act.update_stats(action)
        return self.mean_act.mean

    def _step_init(self, state, action):
        self.external_actuator.initialize_internal_states(state, action)

    def _compute_action(self, action):
        action = self.external_actuator.external_stimulus_to_joint_torques(
            action
        )

        return action

    def _simulation_post_step(self):
        grf = np.concatenate(
            [self._get_collision_force("floor", "right_foot")[:3],
             self._get_collision_force("floor", "left_foot")[:3]]
        )

        self.mean_grf.update_stats(grf)

    def _step_finalize(self):
        self.goal_reward.update_state()
        self.external_actuator.update_state()

    def _get_body_center_of_mass_pos(self, body_name):
        return self._sim.data.subtree_com[
            self._sim.model._body_name2id[body_name]]
