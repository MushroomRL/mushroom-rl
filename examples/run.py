import argparse

from PyPi.agent import Agent
from PyPi import algorithms as algs
from PyPi import approximators as apprxs
from PyPi import environments as envs
from PyPi import policy as pi
from PyPi.utils import logger as l


parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str,
                    help='The name of the environment to solve.')
parser.add_argument('algorithm', type=str,
                    help='The name of the algorithm to run.')
parser.add_argument('--action-regression', action='store_true',
                    help='If true, a separate regressor for each action'
                         'is used.')
parser.add_argument('--logging', default=1, type=int, help='Logging level')
args = parser.parse_args()

# Logger
l.Logger(args.logging)

# MDP
mdp = envs.CarOnHill()

# Spaces
state_space = mdp.observation_space
action_space = mdp.action_space

# Policy
epsilon = 1
policy = pi.EpsGreedy(epsilon)

# Regressor
discrete_actions = mdp.action_space.values
apprx_params = dict(n_input=2, n_output=1, hidden_neurons=[10])
approximator = apprxs.Regressor(approximator_class=apprxs.DenseNN,
                                **apprx_params)
if args.action_regression:
    approximator = apprxs.ActionRegressor(approximator, discrete_actions)

# Agent
agent = Agent(approximator, policy, discrete_actions=discrete_actions)

# Algorithm
fit_params = dict(nb_epoch=20, batch_size=100)
alg_params = dict(gamma=mdp.gamma,
                  learning_rate=1,
                  fit_params=fit_params)
alg = algs.FQI(agent, mdp, **alg_params)
#alg = algs.QLearning(agent, mdp, **alg_params)

# Train
alg.learn(how_many=1000, n_fit_steps=20)
#alg.learn(500)

# Test
agent.policy.set_epsilon(0)

import numpy as np

initial_states = np.zeros((289, 2))
cont = 0
for i in range(-8, 9):
    for j in range(-8, 9):
        initial_states[cont, :] = [0.125 * i, 0.375 * j]
        cont += 1
print(np.mean(alg.evaluate(initial_states)))
