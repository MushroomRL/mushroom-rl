from mushroom_rl.environments import GridWorld

mdp = GridWorld.from_size(width=3, height=3, goal=(2, 2), start=(0, 0))

from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

epsilon = Parameter(value=1.)
policy = EpsGreedy(epsilon=epsilon)

from mushroom_rl.algorithms.value import QLearning

learning_rate = Parameter(value=.6)
agent = QLearning(mdp.info, policy, learning_rate)

from mushroom_rl.core import Core

core = Core(agent, mdp)
core.learn(n_steps=10000, n_steps_per_fit=1)

import numpy as np

shape = agent.Q.shape
q = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        state = np.array([i])
        action = np.array([j])
        q[i, j] = agent.Q.predict(state, action)
print(q)
