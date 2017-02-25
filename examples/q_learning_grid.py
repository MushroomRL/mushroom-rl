from PyPi.agent import Agent
from PyPi import algorithms as algs
from PyPi import approximators as apprxs
from PyPi import environments as envs
from PyPi import policy as pi


mdp = envs.GridWorld(3, 3, (2, 2))
state_space = mdp.observation_space
action_space = mdp.action_space

epsilon = 1
policy = pi.EpsGreedy(epsilon)

discrete_actions = mdp.action_space.values
apprx_params = dict(shape=(3, 3))
approximator = apprxs.Regressor(approximator_class=apprxs.Tabular,
                                **apprx_params)
approximator = apprxs.ActionRegressor(approximator, discrete_actions)

agent = Agent(approximator, policy, discrete_actions=discrete_actions)

alg_params = dict(gamma=mdp.gamma,
                  learning_rate=1)
algorithm = algs.QLearning(agent, mdp, **alg_params)

algorithm.run(50)
