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
        ns_test = np.array([-0.26244546, -2.33917271, 0.50130095, -0.50937527,
                            0.55561752, -0.21111919, -0.55516933, -2.03929087,
                            -18.22893801, 5.89523326, 22.07483625, -2.21756007,
                            3.95695223, 0., 0.])

        assert np.allclose(ns, ns_test)
except ImportError:
    pass
