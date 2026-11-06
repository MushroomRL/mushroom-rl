import numpy as np
import pytest

from helper.envs import rollout

pytest.importorskip("dm_control")

from mushroom_rl.environments.dm_control_env import DMControl  # noqa: E402


def test_dm_control():
    np.random.seed(1)
    mdp = DMControl('hopper', 'hop', 1000, .99, task_kwargs={'random': 1})
    ns = rollout(mdp, lambda mdp: np.random.rand(mdp.info.action_space.shape[0]))
    ns_test = np.load('tests/environments/data/test_dm_control.npy')

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

    rewards_test = np.load('tests/environments/data/test_dm_control_action_repeat_rewards.npy')
    ns_test = np.load('tests/environments/data/test_dm_control_action_repeat_ns.npy')

    assert np.allclose(rewards, rewards_test)
    assert np.allclose(ns, ns_test)

    with pytest.raises(AssertionError):
        DMControl('walker', 'stand', 1000, .99, action_repeat=0)


def test_interface():
    mdp = DMControl('cartpole', 'balance', None, .99, task_kwargs={'random': 1})
    state, _ = mdp.reset()

    assert mdp.info.horizon == 1000
    assert mdp.full_name() == 'DMControl.cartpole.balance'

    with pytest.raises(NotImplementedError):
        mdp.reset(state)
