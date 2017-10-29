from mushroom.algorithms.agent import Agent


class PolicyGradient(Agent):

    def __init__(self, policy, gamma, params, features):
        self.learning_rate = params['algorithm_params'].pop('learning_rate')
        self.J_episode = None
        self.df = None

        super(PolicyGradient, self).__init__(policy, gamma, params, features)

    def fit(self, dataset, n_iterations):
        """
        Fit loop.

        Args:
            dataset (list): the dataset;
            n_iterations (int): number of iterations.

        Returns:
            Last target computed.

        """
        assert n_iterations == 1

        J = list()
        df = 1.
        J_episode = 0.
        self._init_update()
        for sample in dataset:
            state, action, reward, next_state, _, last = self._parse(sample)
            J_episode += df * reward
            df *= self._gamma
            self._step_update(state, action)

            if last:
                self._episode_end_update(J_episode)
                J.append(J_episode)
                J_episode = 0
                df = 1
                self._init_update()

        grad_J = self._compute_gradient(J)
        theta = self.policy.get_weights()
        theta_new = theta + self.learning_rate() * grad_J

        self.policy.set_weights(theta_new)

    def _init_update(self):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _compute_gradient(self, J):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _step_update(self, state, action):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _episode_end_update(self, J_episode):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _parse(self, sample):
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        last = sample[5]

        if self.phi is not None:
            state = self.phi(state)

        return state, action, reward, next_state, absorbing, last

    def __str__(self):
        return self.__name__
