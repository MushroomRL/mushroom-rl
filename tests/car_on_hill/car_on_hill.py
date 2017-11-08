import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

from mushroom.algorithms.value.batch_td import FQI
from mushroom.approximators import Regressor
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter


def experiment(boosted):
    np.random.seed(20)

    # MDP
    mdp = CarOnHill()

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    if not boosted:
        approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                                   n_actions=mdp.info.action_space.n,
                                   params={'n_estimators': 50,
                                           'min_samples_split': 5,
                                           'min_samples_leaf': 2})
    else:
        approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                                   n_actions=mdp.info.action_space.n,
                                   n_models=3,
                                   prediction='sum',
                                   params={'n_estimators': 50,
                                           'min_samples_split': 5,
                                           'min_samples_leaf': 2})

    approximator = ExtraTreesRegressor

    # Agent
    algorithm_params = dict(boosted=boosted, quiet=True)
    fit_params = dict()
    agent_params = {'approximator_params': approximator_params,
                    'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = FQI(approximator, pi, mdp.info, agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=1, how_many=50, n_fit_steps=3,
               iterate_over='episodes', quiet=True)
    core.reset()

    # Test
    test_epsilon = Parameter(0)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.zeros((9, 2))
    cont = 0
    for i in range(-8, 9, 8):
        for j in range(-8, 9, 8):
            initial_states[cont, :] = [0.125 * i, 0.375 * j]
            cont += 1

    dataset = core.evaluate(initial_states=initial_states, quiet=True)

    return np.mean(compute_J(dataset, mdp.info.gamma))


if __name__ == '__main__':
    print('Executing car_on_hill test...')

    n_experiment = 2

    Js = Parallel(n_jobs=-1)(
        delayed(experiment)(False) for _ in range(n_experiment))
    assert np.round(np.mean(Js), 5) == -0.41931
    Js = Parallel(n_jobs=-1)(
        delayed(experiment)(True) for _ in range(n_experiment))
    assert np.round(np.mean(Js), 5) == -0.43776
