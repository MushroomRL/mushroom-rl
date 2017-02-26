import logging

from PyPi.agent import Agent
from PyPi import algorithms as algs
from PyPi import approximators as apprxs
from PyPi import environments as envs
from PyPi import policy as pi

# Logger
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

# MDP
mdp = envs.GridWorld(3, 3, (2, 2))

# Spaces
state_space = mdp.observation_space
action_space = mdp.action_space

# Policy
epsilon = .5
policy = pi.EpsGreedy(epsilon)

# Regressor
discrete_actions = mdp.action_space.values
apprx_params = dict(shape=(3, 3))
approximator = apprxs.Regressor(approximator_class=apprxs.Tabular,
                                **apprx_params)
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
print(algorithm.run(10, True))
