import time

import numpy as np
from mushroom_rl.environments.pybullet_envs.air_hockey.single import  AirHockeySingle


class AirHockeyPrepare(AirHockeySingle):
    def __init__(self, gamma=0.99, horizon=500, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 random_init=False, action_penalty=1e-3):

        self.random_init = random_init
        self.action_penalty = action_penalty

        self.start_range = np.array([[-0.6, -0.3], [0.25, 0.4]])

        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function)

    def setup(self, state):
        if self.random_init:
            puck_pos = np.random.rand(2) * ()

        else:
            puck_pos = np.mean(self.start_range, axis=1)

        puck_pos = np.concatenate([puck_pos, [-0.189]])
        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])

        for i, (model_id, joint_id, _) in enumerate(self._indexer.action_data):
            self.client.resetJointState(model_id, joint_id, self.init_state[i])

    def reward(self, state, action, next_state, absorbing):
        r = 0

        return r

    def _simulation_post_step(self):
        pass


if __name__ == '__main__':
    env = AirHockeyPrepare(debug_gui=True, obs_noise=False, obs_delay=False, n_intermediate_steps=4)

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