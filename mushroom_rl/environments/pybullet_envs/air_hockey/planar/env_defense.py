import time

import numpy as np

from mushroom_rl.core import MDPInfo
from mushroom_rl.environments.pybullet_envs.air_hockey.planar.env_single import AirHockeyPlanarSingle, \
    PyBulletObservationType
from mushroom_rl.utils.spaces import Box


class AirHockeyPlanarDefense(AirHockeyPlanarSingle):
    def __init__(self, seed=None, gamma=0.99, horizon=500, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, control_type="torque",
                 random_init=False, step_action_function=None, action_penalty=1e-3):
        self.start_range = np.array([[0.2, 0.78], [-0.4, 0.4]])
        self.has_hit = False
        self.has_bounce = False
        self.random_init = random_init
        self.action_penalty = action_penalty
        super().__init__(seed=seed, gamma=gamma, horizon=horizon, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay, control_type=control_type,
                         step_action_function=step_action_function)
        self.init_state = np.array([-1.1, 0.8, np.pi/2])

    def setup(self, state=None):
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]
            puck_pos = np.concatenate([puck_pos, [-0.189]])
            puck_lin_vel = np.random.uniform(-1, 1, 3) * 0.5
            puck_lin_vel[0] = -1.0
            puck_lin_vel[2] = 0.0
            puck_ang_vel = np.random.uniform(-1, 1, 3)
            puck_ang_vel[:2] = 0.0
        else:
            puck_pos = np.array([self.start_range[0].mean(), 0.0])
            puck_pos = np.concatenate([puck_pos, [-0.189]])
            puck_lin_vel = np.array([-1., 0., 0.])
            puck_ang_vel = np.zeros(3)

        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])
        self.client.resetBaseVelocity(self._model_map['puck'], puck_lin_vel, puck_ang_vel)

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self._client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_hit = False
        self.has_bounce = False

    def _modify_mdp_info(self, mdp_info):
        obs_idx = [0, 1, 2, 7, 8, 9, 13, 14, 15, 16, 17, 18]
        obs_low = mdp_info.observation_space.low[obs_idx]
        obs_high = mdp_info.observation_space.high[obs_idx]
        obs_low[2] = - np.pi
        obs_high[2] = np.pi
        observation_space = Box(low=obs_low, high=obs_high)
        return MDPInfo(observation_space, mdp_info.action_space, mdp_info.gamma, mdp_info.horizon)

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_POS)[:3]
        if absorbing:
            if puck_pos[0] + self.env_spec['table']['length'] / 2 < 0 and \
                    np.abs(puck_pos[1]) - self.env_spec['table']['goal'] < 0:
                r = -30
        else:
            if self.has_bounce:
                r = 0
            elif puck_pos[0] > -0.8:
                if not self.has_hit:
                    r = 5 * np.exp(-20 * np.abs(puck_pos[0] + 0.6)) + 0.5
                else:
                    ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                                PyBulletObservationType.LINK_POS)[:2]
                    dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos[:2])
                    r = np.exp(-10 * dist_ee_puck)
            else:
                r = -5

        r -= self.action_penalty * np.linalg.norm(action)
        return r

    def is_absorbing(self, state):
        if super().is_absorbing(state):
            return True
        return False

    def _simulation_post_step(self):
        if not self.has_hit:
            collision_count = len(self.client.getContactPoints(self._model_map['puck'],
                                                               self._indexer.link_map['planar_robot_1/'
                                                                                      'link_striker_ee'][0],
                                                               -1,
                                                               self._indexer.link_map['planar_robot_1/'
                                                                                      'link_striker_ee'][1]))
            if collision_count > 0:
                self.has_hit = True

        if not self.has_bounce:
            collision_count = 0
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_up_rim_l'][0],
                                                                -1,
                                                                self._indexer.link_map['t_up_rim_l'][1]))
            collision_count += len(self.client.getContactPoints(self._model_map['puck'],
                                                                self._indexer.link_map['t_up_rim_r'][0],
                                                                -1,
                                                                self._indexer.link_map['t_up_rim_r'][1]))
            if collision_count > 0:
                self.has_bounce = True


if __name__ == '__main__':
    env = AirHockeyPlanarDefense(debug_gui=True, obs_noise=False, obs_delay=False)

    while True:
        action = np.random.randn(3) * 10
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        time.sleep(0.01)
