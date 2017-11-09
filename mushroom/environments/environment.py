class MDPInfo:
    def __init__(self, observation_space, action_space, gamma, horizon):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

    @property
    def size(self):
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        return self.observation_space.shape + self.action_space.shape


class Environment(object):
    def __init__(self, mdp_info):
        # MDP initialization
        self.reset()

        self._mdp_info = mdp_info

    def seed(self, seed):
        if hasattr(self, 'env'):
            self.env.seed(seed)
        else:
            raise NotImplementedError

    @property
    def info(self):
        return self._mdp_info

    def __str__(self):
        return self.__name__
