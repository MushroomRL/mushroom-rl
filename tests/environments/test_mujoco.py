try:
    from mushroom_rl.environments.dm_control_env import DMControl
    import numpy as np

    def test_dm_control():
        np.random.seed(1)
        mdp = DMControl('hopper', 'hop', 1000, .99, task_kwargs={'random': 1})
        mdp.reset()
        for i in range(10):
            ns, r, ab, _ = mdp.step(
                np.random.rand(mdp.info.action_space.shape[0]))
        ns_test = np.array([-0.25868173,  -2.24011367,   0.45346572,  -0.55528368,
                            0.51603826,  -0.21782316,  -0.58708578,  -2.04541986,
                            -17.24931206,   5.42227781,  21.39084468,  -2.42071806,
                            3.85448837,   0.,   0.])

        assert np.allclose(ns, ns_test)
except ImportError:
    pass
