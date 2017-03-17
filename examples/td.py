import argparse
import json

from PyPi.agent import Agent
from PyPi.utils import logger
from PyPi.utils.loader import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='The path of the experiment'
                                               'configuration file.')
parser.add_argument('--logging', default=1, type=int, help='Logging level.')
args = parser.parse_args()

# Load config file
if args.config is not None:
    load_path = args.config
    with open(load_path) as f:
        config = json.load(f)
else:
    raise ValueError('Configuration file path missing.')

# Logger
logger.Logger(args.logging)

# MDP
mdp = get_environment(config['environment']['name'],
                      **config['environment']['params'])

# Spaces
state_space = mdp.observation_space
action_space = mdp.action_space

# Policy
policy = get_policy(config['policy']['name'],
                    **config['policy']['params'])

# Regressor
discrete_actions = mdp.action_space.values
approximator = get_approximator(config['approximator']['name'],
                                **config['approximator']['params'])
if config['approximator']['action_regression']:
    approximator = apprxs.ActionRegressor(approximator, discrete_actions)

# Agent
agent = Agent(approximator, policy, discrete_actions=discrete_actions)

# Algorithm
alg = get_algorithm(config['algorithm']['name'],
                    agent,
                    mdp,
                    **config['algorithm']['params'])

# Train
alg.learn(config['algorithm']['how_many'], config['algorithm']['n_fit_steps'])

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
