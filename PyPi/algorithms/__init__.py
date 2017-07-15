from .batch_td import BatchTD, FQI, DoubleFQI, WeightedFQI
from .dqn import DQN
from .td import TD, QLearning, DoubleQLearning, WeightedQLearning, SARSA

__all__ = ['Algorithm', 'BatchTD', 'FQI', 'DQN', 'DoubleFQI', 'WeightedFQI',
           'TD', 'QLearning', 'DoubleQLearning', 'WeightedQLearning', 'SARSA']
