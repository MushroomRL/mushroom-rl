try:
    from mushroom_rl.environments.mujoco_envs.air_hockey import AirHockeyDefend, AirHockeyHit, AirHockeyPrepare,\
        AirHockeyRepel
    import numpy as np

    def test_defend():
        obs = []
        mdp = AirHockeyDefend()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/air_hockey_defend_data.npy")

        assert np.allclose(obs, obs_test)


    def test_hit():
        obs = []
        mdp = AirHockeyHit()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/air_hockey_hit_data.npy")

        assert np.allclose(obs, obs_test)


    def test_prepare():
        obs = []
        mdp = AirHockeyPrepare()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/air_hockey_prepare_data.npy")

        assert np.allclose(obs, obs_test)


    def test_repel():
        obs = []
        mdp = AirHockeyRepel()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/air_hockey_repel_data.npy")

        assert np.allclose(obs, obs_test)


except ImportError:
    pass
