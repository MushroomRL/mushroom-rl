import numpy as np

from conv_net import AtariConvNet
from PyPi.agent import Agent
from PyPi.utils.loader import *
from PyPi.utils.parameters import Parameter


mdp, policy, _, config = load_experiment()
approximator_params = dict(n_actions=mdp.action_space.n_values)
approximator = apprxs.Regressor(AtariConvNet, **approximator_params)

# Agent
discrete_actions = mdp.action_space.values
agent = Agent(approximator, policy, discrete_actions=discrete_actions)

# Algorithm
alg = get_algorithm(config['algorithm']['name'],
                    agent,
                    mdp,
                    **config['algorithm']['params'])

# Train
alg.learn(n_iterations=np.inf, how_many=1, n_fit_steps=1,
          iterate_over='samples', initial_dataset_size=10, render=True)

# Test
agent.policy.set_epsilon(Parameter(0))
