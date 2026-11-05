import numpy as np
import pytest

from mushroom_rl.core import MultiprocessEnvironment
from mushroom_rl.core import multiprocess_environment
from mushroom_rl.environments import Gymnasium, LQR, Segway


def reset_states(env, seed, n_envs=3, **kwargs):
    mdp = MultiprocessEnvironment(env, n_envs=n_envs, **kwargs)
    if seed is not None:
        mdp.seed(seed)
    states, _ = mdp.reset_all(np.ones(n_envs, dtype=bool))
    mdp.close_all()

    return states


def assert_distinct(states):
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            assert not np.array_equal(states[i], states[j])


def test_seed():
    pytest.importorskip('gymnasium')
    first_run = reset_states(Gymnasium, 5, name='Pendulum-v1', horizon=200, gamma=.99)

    assert np.array_equal(first_run, reset_states(Gymnasium, 5, name='Pendulum-v1', horizon=200, gamma=.99))
    assert not np.array_equal(first_run, reset_states(Gymnasium, 6, name='Pendulum-v1', horizon=200, gamma=.99))
    assert not np.array_equal(reset_states(Gymnasium, None, name='Pendulum-v1', horizon=200, gamma=.99),
                              reset_states(Gymnasium, None, name='Pendulum-v1', horizon=200, gamma=.99))
    assert_distinct(first_run)


def test_seed_global_generator():
    first_run = reset_states(Segway, 5, random_start=True)

    assert np.array_equal(first_run, reset_states(Segway, 5, random_start=True))
    assert not np.array_equal(first_run, reset_states(Segway, 6, random_start=True))

    np.random.seed(3)
    assert_distinct(reset_states(Segway, None, random_start=True))


def test_reset_and_step():
    initial_states = np.array([[1., 2.], [3., 4.], [5., 6.]])
    env_mask = np.array([True, False, True])
    mdp = MultiprocessEnvironment(LQR, dimensions=2, n_envs=3, use_generator=True)

    states, _ = mdp.reset_all(np.ones(3, dtype=bool), initial_states)

    assert np.array_equal(states, initial_states)

    states, _ = mdp.reset_all(env_mask, initial_states[::-1])

    assert np.array_equal(states[env_mask], initial_states[::-1][env_mask])

    mdp.set_default_env(1)
    initial_state, _ = mdp.reset(np.array([1., 2.]))
    state, reward, absorbing, _ = mdp.step(np.array([10., 20.]))
    mdp.close_all()

    assert np.array_equal(initial_state, np.array([1., 2.]))
    assert np.array_equal(state, np.array([11., 22.]))
    assert reward == -371.3
    assert not absorbing


def test_construction(monkeypatch):
    monkeypatch.setattr(multiprocess_environment, 'cpu_count', lambda: 2)
    mdp = MultiprocessEnvironment.generate(LQR, dimensions=2)

    assert mdp.number == 2
    assert mdp.full_name() == 'MultiprocessEnvironment.LQR'
    assert mdp.info.shape == (2, 2)

    mdp.close_all()
    mdp.close_all()

    mdp = MultiprocessEnvironment.generate(Segway, n_envs=2)

    assert mdp.full_name() == 'MultiprocessEnvironment.Segway'

    mdp.close_all()

    with pytest.raises(AssertionError):
        MultiprocessEnvironment(LQR, n_envs=1)


def test_render():
    mdp = MultiprocessEnvironment(Segway, n_envs=2)
    mdp.reset_all(np.ones(2, dtype=bool))

    frames = mdp.render_all(np.array([True, False]), record=True)
    frame = mdp.render(record=True)

    assert mdp.render_all(np.ones(2, dtype=bool)) is None
    assert frames.ndim == 4 and frames.shape[0] == 1 and frames.shape[-1] == 3
    assert frame.shape == frames.shape[1:]

    mdp.stop()
    mdp.close_all()
