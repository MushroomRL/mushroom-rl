from .td import TD
from .sarsa import SARSA, SARSALambda, ExpectedSARSA
from .q_learning import QLearning, DoubleQLearning, SpeedyQLearning
from .r_learning import RLearning
from .weighted_q_learning import WeightedQLearning
from .rq_learning import RQLearning
from .sarsa_continuous import SARSALambdaContinuous, TrueOnlineSARSALambda

__all__ = ['SARSA', 'SARSALambda', 'ExpectedSARSA',
           'QLearning', 'DoubleQLearning', 'SpeedyQLearning',
           'RLearning', 'WeightedQLearning', 'RQLearning',
           'SARSALambdaContinuous', 'TrueOnlineSARSALambda']