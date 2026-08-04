import numpy as np

from mushroom_rl.core import MultiprocessEnvironment
from mushroom_rl.environments import Gymnasium, LQR, Segway


def reset_states(seed, n_envs):
    mdp = MultiprocessEnvironment(Gymnasium, 'Pendulum-v1', 200, .99, n_envs=n_envs)
    mdp.seed(seed)
    states, _ = mdp.reset_all(np.ones(n_envs, dtype=bool))
    mdp.close_all()

    return states


def reset_states_global_generator(seed, n_envs):
    mdp = MultiprocessEnvironment(Segway, random_start=True, n_envs=n_envs)

    if seed is not None:
        mdp.seed(seed)

    states, _ = mdp.reset_all(np.ones(n_envs, dtype=bool))
    mdp.close_all()

    return states


def test_multiprocess_environment_seed():
    first_run = reset_states(5, 3)
    second_run = reset_states(5, 3)
    other_run = reset_states(6, 3)

    assert np.array_equal(first_run, second_run)
    assert not np.array_equal(first_run, other_run)


def test_multiprocess_environment_seed_differs_per_copy():
    states = reset_states(5, 3)

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            assert not np.array_equal(states[i], states[j])


def test_multiprocess_environment_seed_none():
    first_run = reset_states(None, 3)
    second_run = reset_states(None, 3)

    assert not np.array_equal(first_run, second_run)


def test_multiprocess_environment_seed_global_generator():
    first_run = reset_states_global_generator(5, 3)
    second_run = reset_states_global_generator(5, 3)
    other_run = reset_states_global_generator(6, 3)

    assert np.array_equal(first_run, second_run)
    assert not np.array_equal(first_run, other_run)


def test_multiprocess_environment_global_generator_differs_per_copy():
    np.random.seed(3)
    states = reset_states_global_generator(None, 3)

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            assert not np.array_equal(states[i], states[j])


def test_multiprocess_environment_reset_all_initial_states():
    n_envs = 3
    initial_states = np.array([[1., 2.], [3., 4.], [5., 6.]])

    mdp = MultiprocessEnvironment(LQR, dimensions=2, n_envs=n_envs, use_generator=True)
    states, _ = mdp.reset_all(np.ones(n_envs, dtype=bool), initial_states)
    mdp.close_all()

    assert np.array_equal(states, initial_states)


def test_multiprocess_environment_reset_all_initial_states_masked():
    n_envs = 3
    initial_states = np.array([[1., 2.], [3., 4.], [5., 6.]])
    env_mask = np.array([True, False, True])

    mdp = MultiprocessEnvironment(LQR, dimensions=2, n_envs=n_envs, use_generator=True)
    states, _ = mdp.reset_all(env_mask, initial_states)
    mdp.close_all()

    assert np.array_equal(states[env_mask], initial_states[env_mask])


def test_multiprocess_environment_default_env_interface():
    n_envs = 3
    default_env = 1

    mdp = MultiprocessEnvironment(LQR, dimensions=2, n_envs=n_envs, use_generator=True)
    mdp.set_default_env(default_env)

    initial_states, _ = mdp.reset(np.array([1., 2.]))
    states, rewards, absorbing, _ = mdp.step(np.array([10., 20.]))
    mdp.close_all()

    assert np.array_equal(initial_states[default_env], np.array([1., 2.]))
    assert np.array_equal(states[default_env], np.array([11., 22.]))
    assert rewards[default_env] == -371.3
    assert not absorbing[default_env]
