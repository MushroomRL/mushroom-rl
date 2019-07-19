from mushroom.environments.ball_in_a_cup import BallInACup
import numpy as np


def linear_movement(start, end, n_steps, i):
    t = np.minimum(1., float(i) / float(n_steps))
    return start + (end - start) * t


def main():
    env = BallInACup()

    des_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
    p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
    d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

    obs = env.reset()
    done = False
    i = 0
    while not done:
        a = p_gains * (linear_movement(des_pos, np.zeros_like(des_pos), 500, i) - obs[0:14:2]) + d_gains * (np.zeros_like(des_pos) - obs[1:14:2])

        # Check the observations
        assert np.all(obs[0:14:2] - env.sim.data.qpos[0:7] == 0)
        assert np.all(obs[1:14:2] - env.sim.data.qvel[0:7] == 0)
        if not np.all(obs[14:17] - env.sim.data.body_xpos[40] == 0):
            print(obs[14:17] - env.sim.data.body_xpos[40] )
        if not np.all(obs[17:] - env.sim.data.body_xvelp[40] == 0):
            print(obs[17:] - env.sim.data.body_xvelp[40])

        obs, reward, done, info = env.step(a)
        env.render()
        i += 1


if __name__ == "__main__":
    main()
