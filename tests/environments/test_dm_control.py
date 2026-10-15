import pytest

pytest.importorskip("mushroom_rl.environments.dm_control_env")

from mushroom_rl.environments.dm_control_env import DMControl
import numpy as np


def test_dm_control():
    np.random.seed(1)
    mdp = DMControl('hopper', 'hop', 1000, .99, task_kwargs={'random': 1})
    mdp.reset()
    states, snapshots = [], []
    for i in range(10):
        ns, r, ab, _ = mdp.step(
            np.random.rand(mdp.info.action_space.shape[0]))
        states.append(ns)
        snapshots.append(np.copy(ns))
    for state, snapshot in zip(states, snapshots):
        assert np.allclose(state, snapshot)
    ns_test = np.array([-0.25868173,  -2.24011367,   0.45346572,  -0.55528368,
                        0.51603826,  -0.21782316,  -0.58708578,  -2.04541986,
                        -17.24931206,   5.42227781,  21.39084468,  -2.42071806,
                        3.85448837,   0.,   0.])

    assert np.allclose(ns, ns_test)


def test_dm_control_action_repeat():
    np.random.seed(1)
    mdp = DMControl('walker', 'stand', 1000, .99, task_kwargs={'random': 1}, action_repeat=3)
    mdp_single = DMControl('walker', 'stand', 1000, .99, task_kwargs={'random': 1})

    state, _ = mdp.reset()
    state_single, _ = mdp_single.reset()

    assert np.array_equal(state, state_single)

    rewards = list()
    for i in range(5):
        action = np.random.rand(mdp.info.action_space.shape[0])
        ns, r, ab, _ = mdp.step(action)

        r_single = 0.
        for _ in range(3):
            ns_single, r_step, ab_single, _ = mdp_single.step(action)
            r_single += r_step

        assert np.array_equal(ns, ns_single)
        assert r == r_single
        assert not ab and not ab_single

        rewards.append(r)

    rewards_test = np.array([2.97383158, 2.99797390, 2.95410732, 2.58159748, 1.77065737])
    ns_test = np.array([0.65798677, 0.75302949, 0.60356003, -0.79731756, 0.59675129, -0.80242626,
                        0.55448362, -0.83219464, 0.63680442, -0.77102537, 0.5918498, -0.80604827,
                        -0.11171683, -0.99374008, 0.85471507, -0.98419927, -1.90334901, -3.24994542,
                        -0.92198224, 0.0464916, -13.45926266, 0.35374391, -1.98270003, -5.2106264])

    assert np.allclose(rewards, rewards_test)
    assert np.allclose(ns, ns_test)


def test_dm_control_action_repeat_invalid():
    with pytest.raises(AssertionError):
        DMControl('walker', 'stand', 1000, .99, action_repeat=0)
