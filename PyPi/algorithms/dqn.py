from PyPi.algorithms import Algorithm


class DQN(Algorithm):
    def __init__(self):
        """
        Deep Q-Network (DQN) algorithm.
        "Human-Level Control through Deep Reinforcement Learning", Mnih V. et.al.. 2015.
        """
        self.__name__ = 'DQN'

        pass

    def __str__(self):
        return self.__name__
