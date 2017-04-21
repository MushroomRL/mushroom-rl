import numpy as np
from joblib import Parallel, delayed

from PyPi.agent import Agent
from PyPi.utils.loader import *
from PyPi.utils.parameters import Parameter


def experiment():
    np.random.seed()

    mdp, policy, approximator, config = load_experiment()

    # Agent
    #discrete_actions = np.linspace(
    #    mdp.action_space.low, mdp.action_space.high, 5).reshape(-1, 1)
    discrete_actions = mdp.action_space.values
    agent = Agent(approximator, policy, discrete_actions=discrete_actions)

    # Algorithm
    #from keras.callbacks import EarlyStopping
    #config['algorithm']['params']['fit_params']['callbacks'] =\
    #    [EarlyStopping('loss', patience=20, min_delta=1e-4)]
    alg = get_algorithm(config['algorithm']['name'],
                        agent,
                        mdp,
                        **config['algorithm']['params'])

    # Train
    alg.learn(n_iterations=1, how_many=1000, n_fit_steps=20,
              iterate_over='episodes')

    # Test
    agent.policy.set_epsilon(Parameter(0))

    initial_states = np.zeros((289, 2))
    cont = 0
    for i in range(-8, 9):
        for j in range(-8, 9):
            initial_states[cont, :] = [0.125 * i, 0.375 * j]
            cont += 1

    return np.mean(alg.evaluate(initial_states, render=False))


if __name__ == '__main__':
    n_experiment = 3

    Js = Parallel(n_jobs=-1)(delayed(experiment)() for _ in range(n_experiment))
    print(Js)
