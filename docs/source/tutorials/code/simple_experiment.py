import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core
from mushroom_rl.environments import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

mdp = CarOnHill()

# Policy
epsilon = Parameter(value=1.)
pi = EpsGreedy(epsilon=epsilon)

# Approximator
approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                           n_actions=mdp.info.action_space.n,
                           n_estimators=50,
                           min_samples_split=5,
                           min_samples_leaf=2)
approximator = ExtraTreesRegressor

# Agent
agent = FQI(mdp.info, pi, approximator, n_iterations=20,
            approximator_params=approximator_params)

core = Core(agent, mdp)

core.learn(n_episodes=1000, n_episodes_per_fit=1000)

pi.set_epsilon(Parameter(0.))
initial_state = np.array([[-.5, 0.]])
dataset = core.evaluate(initial_states=initial_state)

print(dataset.discounted_return)
