import argparse

from PyPi.agent import Agent
from PyPi import algorithms as algs
from PyPi import approximators as apprxs
from PyPi import environments as envs
from PyPi import policy as pi
from PyPi.utils import loader
from PyPi.utils import logger as l


parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str,
                    help='The name of the environment to solve.')
parser.add_argument('algorithm', type=str,
                    help='The name of the algorithm to run.')
parser.add_argument('--train-episodes', type=int, default=100,
                    help='Number of train episodes.')
parser.add_argument('--test-episodes', type=int, default=10,
                    help='Number of test episodes.')
parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
parser.add_argument('--action-regression', action='store_true',
                    help='If true, a separate regressor for each action'
                         'is used.')
parser.add_argument('--logging', default=1, type=int, help='Logging level')
args = parser.parse_args()

args = loader.load(args)

# Logger
logger = l.Logger(args.logging)

# MDP
mdp = envs.GridWorld(8, 8, (7, 7))

# Spaces
state_space = mdp.observation_space
action_space = mdp.action_space

# Policy
epsilon = .5
policy = pi.EpsGreedy(epsilon)

# Regressor
discrete_actions = mdp.action_space.values
apprx_params = dict(shape=(8, 8))
approximator = apprxs.Regressor(approximator_class=apprxs.Tabular,
                                **apprx_params)
if args.action_regression:
    approximator = apprxs.ActionRegressor(approximator, discrete_actions)

# Agent
agent = Agent(approximator, policy, discrete_actions=discrete_actions)

# Algorithm
alg_params = dict(gamma=mdp.gamma,
                  learning_rate=1)
algorithm = algs.QLearning(agent, mdp, logger, **alg_params)

# Train
algorithm.run(50)

# Test
agent.policy.set_epsilon(0)
algorithm.run(10, True)
