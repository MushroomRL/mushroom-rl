from .batch_td import *
from .dqn import *
from .td import *

__all__ = ['FQI', 'DoubleFQI', 'LSPI', 'DQN', 'DoubleDQN',
           'AveragedDQN', 'CategoricalDQN', 'QLearning', 'DoubleQLearning',
           'WeightedQLearning', 'SpeedyQLearning', 'RLearning', 'RQLearning',
           'SARSA', 'SARSALambda', 'SARSALambdaContinuous',
           'ExpectedSARSA', 'TrueOnlineSARSALambda']
