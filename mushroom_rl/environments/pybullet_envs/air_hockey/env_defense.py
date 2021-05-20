import time
import numpy as np
from mushroom_rl.environments.pybullet_envs.air_hockey.env_single import AirHockeyPlanarSingle


class AirHockeyPlanarDefense(AirHockeyPlanarSingle):
    def __init__(self, seed=None, gamma=0.99, horizon=500, timestep=1 / 240., debug_gui=False, env_noise=False,
                 obs_noise=True, obs_delay=False, control_type="torque", random_init=False):
        self.start_range = np.array([[0.2, 0.78], [-0.4, 0.4]])
        self.random_init = random_init
        self.reachable = False
        super().__init__(seed=seed, gamma=gamma, horizon=horizon, timestep=timestep, debug_gui=debug_gui,
                         env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay, control_type=control_type)

    def setup(self, state=None):
        if self.random_init:
            puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]
            puck_pos = np.concatenate([puck_pos, [-0.189]])
            puck_lin_vel = np.random.randn(3) * 0.5
            puck_lin_vel[0] = -1.0
            puck_lin_vel[2] = 0.0
            puck_ang_vel = np.random.randn(3)
            puck_ang_vel[:2] = 0.0
        else:
            puck_pos = np.mean(self.start_range, axis=1)
            puck_pos = np.concatenate([puck_pos, [-0.189]])
            puck_lin_vel = np.array([-1., 0., 0.])
            puck_ang_vel = np.zeros(3)

        self.client.resetBasePositionAndOrientation(self._model_map['puck'], puck_pos, [0, 0, 0, 1.0])
        self.client.resetBaseVelocity(self._model_map['puck'], puck_lin_vel, puck_ang_vel)

        for i, (model_id, joint_id, _) in enumerate(self._action_data):
            self._client.resetJointState(model_id, joint_id, self.init_state[i])

    def reward(self, state, action, next_state, absorbing):
        puck_vel = state[3:6]
        return np.clip(1 - abs(puck_vel[0]), 0, 1)

    def is_absorbing(self, state):
        if super().is_absorbing(state):
            return True
        return False


if __name__ == '__main__':
    env = AirHockeyPlanarDefense(debug_gui=True, obs_noise=False, obs_delay=False)

    while True:
        action = np.random.randn(3) * 10
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        time.sleep(0.01)
