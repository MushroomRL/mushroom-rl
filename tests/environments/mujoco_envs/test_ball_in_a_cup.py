try:
    from mushroom_rl.environments.mujoco_envs import BallInACup
    import numpy as np

    def linear_movement(start, end, n_steps, i):
        t = np.minimum(1., float(i) / float(n_steps))
        return start + (end - start) * t


    def test_ball_in_a_cup():
        env = BallInACup()

        des_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])/2
        d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])/10

        obs = env.reset()
        done = False
        i = 0
        while not done and i < 100:
            q_cmd = linear_movement(env.init_robot_pos, des_pos, 500, i)
            q_curr = obs[0:14:2]
            qdot_cur = obs[1:14:2]
            pos_err = q_cmd - q_curr

            a = p_gains * pos_err - d_gains * qdot_cur

            # Check the observations
            assert np.allclose(obs[0:14:2], env._data.qpos[0:7])
            assert np.allclose(obs[1:14:2], env._data.qvel[0:7])
            assert np.allclose(obs[14:17], env._data.xpos[40])
            assert np.allclose(obs[17:], env._data.cvel[40])
            obs, reward, done, info = env.step(a)

            i += 1

except ImportError:
    pass

test_ball_in_a_cup()
