import numpy as np
import time

from mushroom_rl.environments.pybullet_envs.air_hockey.single import AirHockeySingle, \
    PyBulletObservationType



class AirHockeyRepelle(AirHockeySingle):
    def __init__(self, gamma=0.99, horizon=500, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3):
        self.random_init = random_init
        self.action_penalty = action_penalty

        self.start_range = np.array([[0.2, 0.78], [-0.4, 0.4]])
        self.goal = np.array([0.98, 0])

        self.has_hit = False
        self.vel_hit_x = 0.
        self.r_hit = 0.

        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function)

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
        self.vel_hit_x = 0.
        self.r_hit = 0.

    # Very flawed needs a lot of tuning
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


if __name__ == "__main__":
    env = AirHockeyRepelle(debug_gui=True, env_noise=False, obs_noise=False, obs_delay=False, n_intermediate_steps=4,
                           random_init=True)

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
        time.sleep(1 / 60.)