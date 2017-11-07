from .batch_td import FQI, DoubleFQI, WeightedFQI
from .dqn import DQN, DoubleDQN, AveragedDQN, WeightedDQN
from .td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning,\
    RLearning, RQLearning, SARSA


__all__ = ['FQI', 'DoubleFQI', 'WeightedFQI', 'DQN', 'DoubleDQN', 'WeightedDQN',
           'AveragedDQN', 'QLearning', 'DoubleQLearning', 'WeightedQLearning',
           'SpeedyQLearning', 'RLearning', 'RQLearning', 'SARSA']
