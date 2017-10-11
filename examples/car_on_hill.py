import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

from mushroom.algorithms.batch_td import FQI
from mushroom.approximators import ActionRegressor, Ensemble
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter


def experiment():
    np.random.seed()

    # MDP
    mdp = CarOnHill()

    # Policy
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)

    # Approximator
    approximator_params = dict(n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)
    approximator = ActionRegressor(ExtraTreesRegressor,
                                   discrete_actions=mdp.action_space.n,
                                   **approximator_params)
    approximator = Ensemble(ExtraTreesRegressor,
                            n_models=20,
                            prediction='sum',
                            use_action_regressor=True,
                            discrete_actions=mdp.action_space.n,
                            **approximator_params)

    # Agent
    algorithm_params = dict(boosted=True)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = FQI(approximator, pi, mdp.gamma, agent_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=1, how_many=1000, n_fit_steps=20,
               iterate_over='episodes')
    core.reset()

    # Test
    test_epsilon = Parameter(0)
    agent.policy.set_epsilon(test_epsilon)

    initial_states = np.zeros((289, 2))
    cont = 0
    for i in range(-8, 9):
        for j in range(-8, 9):
            initial_states[cont, :] = [0.125 * i, 0.375 * j]
            cont += 1

    dataset = core.evaluate(initial_states=initial_states)

    return np.mean(compute_J(dataset, mdp.gamma))


if __name__ == '__main__':
    n_experiment = 1

    Js = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))
    print(np.mean(Js))
