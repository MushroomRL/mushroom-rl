import argparse

from keras.callbacks import EarlyStopping

from PyPi.agent import Agent
from PyPi.utils import logger
from PyPi.utils.loader import *


parser = argparse.ArgumentParser()
parser.add_argument('algorithm', type=str,
                    help='The name of the algorithm to run.')
parser.add_argument('approximator', type=str,
                    help='The name of the approximator to use.')
parser.add_argument('environment', type=str,
                    help='The name of the environment to solve.')
parser.add_argument('policy', type=str,
                    help='The name of the policy to use.')
parser.add_argument('--action-regression', action='store_true',
                    help='If true, a separate regressor for each action'
                         'is used.')
parser.add_argument('--logging', default=1, type=int, help='Logging level')
args = parser.parse_args()

# Logger
logger.Logger(args.logging)

# MDP
environment_params = dict()
mdp = get_environment(args.environment, **environment_params)

# Spaces
state_space = mdp.observation_space
action_space = mdp.action_space

# Policy
policy_params = dict(epsilon=1)
policy = get_policy(args.policy, **policy_params)

# Regressor
discrete_actions = mdp.action_space.values
#apprx_params = dict(n_input=2, n_output=1, hidden_neurons=[10], loss='mse',
#                    optimizer='rmsprop')
apprx_params = dict(n_estimators=50, min_samples_split=5, min_samples_leaf=2,
                    criterion='mse', input_scaled=False, output_scaled=False)
approximator = get_approximator(args.approximator, **apprx_params)
if args.action_regression:
    approximator = apprxs.ActionRegressor(approximator, discrete_actions)

# Agent
agent = Agent(approximator, policy, discrete_actions=discrete_actions)

# Algorithm
#es = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20)
#fit_params = dict(nb_epoch=500,
#                  batch_size=100,
#                  validation_split=0.1,
#                  callbacks=[es])
fit_params = dict()
alg_params = dict(agent=agent,
                  mdp=mdp,
                  gamma=mdp.gamma,
                  fit_params=fit_params)
alg = get_algorithm(args.algorithm, **alg_params)

# Train
alg.learn(how_many=1000, n_fit_steps=20)

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
