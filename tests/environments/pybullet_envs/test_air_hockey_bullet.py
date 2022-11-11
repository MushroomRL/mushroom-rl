try:
    from mushroom_rl.environments import AirHockeyDefendBullet, AirHockeyHitBullet, AirHockeyPrepareBullet, AirHockeyRepelBullet
    import numpy as np


    def test_defend():
        obs = []
        mdp = AirHockeyDefendBullet()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/pybullet_envs/air_hockey_defend_data.npy")

        assert np.allclose(obs, obs_test)


    def test_hit():
        obs = []
        mdp = AirHockeyHitBullet()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/pybullet_envs/air_hockey_hit_data.npy")

        assert np.allclose(obs, obs_test)


    def test_prepare():
        obs = []
        mdp = AirHockeyPrepareBullet()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/pybullet_envs/air_hockey_prepare_data.npy")

        assert np.allclose(obs, obs_test)


    def test_repel():
        obs = []
        mdp = AirHockeyRepelBullet()
        mdp.reset()
        action = np.array([1] * 3)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/pybullet_envs/air_hockey_repel_data.npy")

        assert np.allclose(obs, obs_test)


except ImportError:
    pass
