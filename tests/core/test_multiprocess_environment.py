import numpy as np

from mushroom_rl.core import MultiprocessEnvironment
from mushroom_rl.environments import Gymnasium, Segway


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
