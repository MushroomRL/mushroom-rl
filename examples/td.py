import argparse
import json

from PyPi.agent import Agent
from PyPi.utils import logger
from PyPi.utils.loader import *
from PyPi.utils import spaces


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
approximator = get_approximator(config['approximator']['name'],
                                **config['approximator']['params'])
if config['approximator']['action_regression']:
    if isinstance(mdp.action_space, spaces.Discrete) or\
        isinstance(mdp.action_space, spaces.DiscreteValued) or\
            isinstance(mdp.action_space, spaces.MultiDiscrete):
        approximator = apprxs.ActionRegressor(approximator,
                                              mdp.action_space.values)
    else:
        raise ValueError('Action regression cannot be done with continuous'
                         ' action spaces.')

# Agent
import numpy as np
discrete_actions = np.linspace(
    mdp.action_space.low, mdp.action_space.high, 5).reshape(-1, 1)
agent = Agent(approximator, policy, discrete_actions=discrete_actions)

# Algorithm
alg = get_algorithm(config['algorithm']['name'],
                    agent,
                    mdp,
                    **config['algorithm']['params'])

# Train
alg.learn(how_many=500, n_fit_steps=20)

# Test
agent.policy.set_epsilon(0)

'''
initial_states = np.zeros((289, 2))
cont = 0
for i in range(-8, 9):
    for j in range(-8, 9):
        initial_states[cont, :] = [0.125 * i, 0.375 * j]
        cont += 1
'''
initial_states = np.array([[-1, 1]])
print(np.mean(alg.evaluate(initial_states, render=True)))
