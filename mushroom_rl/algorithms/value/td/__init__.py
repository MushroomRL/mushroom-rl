from .td import TD
from .sarsa import SARSA
from .sarsa_lambda import SARSALambda
from .expected_sarsa import ExpectedSARSA
from .q_learning import QLearning
from .q_lambda import QLambda
from .double_q_learning import DoubleQLearning
from .double_sor_q_learning import DoubleSORQLearning
from .speedy_q_learning import SpeedyQLearning
from .r_learning import RLearning
from .weighted_q_learning import WeightedQLearning
from .maxmin_q_learning import MaxminQLearning
from .rq_learning import RQLearning, RQLearningOnPolicy
from .sarsa_lambda_continuous import SARSALambdaContinuous
from .true_online_sarsa_lambda import TrueOnlineSARSALambda

__all__ = ['TD', 'SARSA', 'SARSALambda', 'ExpectedSARSA', 'QLearning',
           'QLambda', 'DoubleQLearning', 'DoubleSORQLearning', 'SpeedyQLearning',
           'RLearning', 'WeightedQLearning', 'MaxminQLearning',
           'RQLearning', 'RQLearningOnPolicy', 'SARSALambdaContinuous', 'TrueOnlineSARSALambda']
