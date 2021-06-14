import time

import numpy as np

from mushroom_rl.environments.pybullet_envs.air_hockey.planar.env_single import AirHockeyPlanarSingle, PyBulletObservationType


class AirHockeyPlanarHit(AirHockeyPlanarSingle):
    def __init__(self, seed=None, gamma=0.99, horizon=120, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, control_type="torque",
                 random_init=False, step_action_function=None, action_penalty=1e-3):
        self.hit_range = np.array([[-0.6, -0.2], [-0.4, 0.4]])
        self.goal = np.array([0.98, 0])
        self.has_hit = False
        self.vel_hit_x = 0.
        self.r_hit = 0.
        self.random_init = random_init
        self.action_penalty = action_penalty
        super().__init__(seed=seed, gamma=gamma, horizon=horizon, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay, control_type=control_type,
                         step_action_function=step_action_function)

    def setup(self, state):
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
        else:
            puck_pos = np.mean(self.hit_range, axis=1)

        puck_pos = np.concatenate([puck_pos, [-0.189]])
        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self.client.resetJointState(model_id, joint_id, self.init_state[i])

        self.has_hit = False
        self.vel_hit_x = 0.
        self.r_hit = 0.

    def reward(self, state, action, next_state, absorbing):
        r = 0
        puck_pos = self.get_sim_state(next_state, "puck", PyBulletObservationType.BODY_POS)[:2]
        if absorbing:
            if puck_pos[0] - self.env_spec['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.env_spec['table']['goal'] < 0:
                r = 150
        else:
            if not self.has_hit:
                ee_pos = self.get_sim_state(next_state, "planar_robot_1/link_striker_ee",
                                            PyBulletObservationType.LINK_POS)[:2]
                dist_ee_puck = np.linalg.norm(puck_pos - ee_pos)

                vec_ee_puck = (puck_pos - ee_pos) / dist_ee_puck
                vec_puck_goal = (self.goal - puck_pos) / np.linalg.norm(self.goal - puck_pos)
                cos_ang = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
                r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang
                self.r_hit = r
            else:
                r = 1 + self.r_hit + self.vel_hit_x * 0.1

        r -= self.action_penalty * np.linalg.norm(action)
        return r

    def is_absorbing(self, state):
        if super().is_absorbing(state):
            return True
        if self.has_hit:
            puck_vel = self.get_sim_state(self._state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:2]
            if np.linalg.norm(puck_vel) < 0.01:
                return True
        return False

    def _simulation_post_step(self):
        if not self.has_hit:
            puck_vel = self.get_sim_state(self._state, "puck", PyBulletObservationType.BODY_LIN_VEL)[:2]
            if np.linalg.norm(puck_vel) > 0.1:
                self.has_hit = True
                self.vel_hit_x = puck_vel[0]


if __name__ == '__main__':
    env = AirHockeyPlanarHit(debug_gui=True, env_noise=False, obs_noise=False, obs_delay=False, n_intermediate_steps=4)

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        action = np.random.randn(3) * 5
        observation, reward, done, info = env.step(action)
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
        time.sleep(1/60.)
