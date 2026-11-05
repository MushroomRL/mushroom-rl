import numpy as np
import pytest

from helper.envs import discrete_action, rollout

pytest.importorskip('gymnasium')

from mushroom_rl.environments.gymnasium_env import Gymnasium  # noqa: E402


def test_gymnasium():
    np.random.seed(1)
    mdp = Gymnasium('Acrobot-v1', 1000, .99)
    mdp.seed(1)
    ns = rollout(mdp, discrete_action)

    assert np.allclose(ns, np.load('tests/environments/data/test_gymnasium.npy'))


def test_reset_state():
    mdp = Gymnasium('MountainCar-v0', 1000, .99)
    mdp.seed(1)
    initial_state = np.array([-.75, -.02])
    state, _ = mdp.reset(initial_state)
    assert np.array_equal(state, initial_state)
    assert not np.shares_memory(state, initial_state)
    for i in range(5):
        ns, r, ab, _ = mdp.step(np.array([2]))
    ns_test = np.array([-0.8097278, -0.0063345535])

    assert np.allclose(ns, ns_test)
    assert r == -1.
    assert not ab
    assert np.array_equal(initial_state, np.array([-.75, -.02]))

    mdp = Gymnasium('CartPole-v1', 1000, .99)
    mdp.seed(1)
    initial_state = np.array([.1, 0., .1, 0.])
    state, _ = mdp.reset(initial_state)
    assert np.array_equal(state, initial_state)
    for i in range(5):
        ns, r, ab, _ = mdp.step(np.array([1]))
    ns_test = np.load('tests/environments/data/test_reset_state.npy')

    assert np.allclose(ns, ns_test)
    assert r == 1.
    assert not ab

    for name in ['Pendulum-v1', 'FrozenLake-v1']:
        mdp = Gymnasium(name, 100, .99)
        state, _ = mdp.reset()
        with pytest.raises(NotImplementedError):
            mdp.reset(state)


def test_render():
    mdp = Gymnasium('Pendulum-v1', 200, .99, headless=True)
    for _ in range(2):
        mdp.seed(1)
        mdp.reset()
        frames = []
        for i in range(3):
            mdp.step(np.array([.5]))
            frames.append(mdp.render(record=True))
            assert np.array_equal(frames[-1], mdp.env.render())
            assert mdp.render(record=False) is None
        mdp.stop()

        assert all(frame.shape == (500, 500, 3) and frame.dtype == np.uint8 for frame in frames)
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])


def test_construction():
    gym = pytest.importorskip('gymnasium')
    mdp = Gymnasium('CartPole-v1', 100, .99, wrappers=[(gym.wrappers.TimeAwareObservation, dict())])

    assert mdp.info.observation_space.shape == (5,)
    assert mdp.full_name() == 'Gymnasium.CartPole-v1'
    assert Gymnasium('CartPole-v1', None, .99).info.horizon == 500

    with pytest.raises(RuntimeError):
        Gymnasium('Blackjack-v1', None, .99)
    with pytest.raises(ValueError):
        Gymnasium('Blackjack-v1', 10, .99)
