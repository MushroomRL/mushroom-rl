try:
    from mushroom_rl.environments.mujoco_envs.reach import Reach
    from mushroom_rl.environments.mujoco_envs.pick import Pick
    from mushroom_rl.environments.mujoco_envs.push import Push
    from mushroom_rl.environments.mujoco_envs.peg_insertion import PegInsertion
    import numpy as np
    import random
    import torch


    def test_reach():
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        obs = []
        mdp = Reach()
        mdp.reset()
        action = np.zeros(7)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/reach_data.npy")

        assert np.allclose(obs, obs_test)


    def test_pick():
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        obs = []
        mdp = Pick()
        mdp.reset()
        action = np.zeros(8)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/pick_data.npy")

        assert np.allclose(obs, obs_test)


    def test_push():
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        obs = []
        mdp = Push()
        mdp.reset()
        action = np.zeros(7)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/push_data.npy")

        assert np.allclose(obs, obs_test)


    def test_peg_insertion():
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)

        obs = []
        mdp = PegInsertion()
        mdp.reset()
        action = np.zeros(7)

        for _ in range(20):
            observation, _, _, _ = mdp.step(action)
            assert len(observation) == len(mdp._mdp_info.observation_space.low)
            assert len(observation) == len(mdp._mdp_info.observation_space.high)
            obs.append(observation)

        obs_test = np.load("tests/environments/mujoco_envs/peg_insertion_data.npy")

        assert np.allclose(obs, obs_test)


except ImportError:
    pass
