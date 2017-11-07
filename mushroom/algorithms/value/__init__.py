from .batch_td import FQI, DoubleFQI, WeightedFQI, LSPI
from .dqn import DQN, DoubleDQN, AveragedDQN, WeightedDQN
from .td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning,\
    RLearning, RQLearning, SARSA, ExpectedSARSA, LinearSARSA


__all__ = ['FQI', 'DoubleFQI', 'WeightedFQI', 'LSPI', 'DQN', 'DoubleDQN',
           'WeightedDQN', 'AveragedDQN', 'QLearning', 'DoubleQLearning',
           'WeightedQLearning', 'SpeedyQLearning', 'RLearning', 'RQLearning',
           'SARSA', 'ExpectedSARSA', 'LinearSARSA']
