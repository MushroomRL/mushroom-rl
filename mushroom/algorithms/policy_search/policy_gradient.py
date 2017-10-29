from mushroom.algorithms.agent import Agent


class PolicyGradient(Agent):

    def __init__(self, policy, gamma, params, features):
        self.learning_rate = params['algorithm_params'].pop('learning_rate')
        self.Jep = None
        self.df = None

        super(PolicyGradient, self).__init__(policy, gamma, params, features)

    def fit(self, dataset, n_iterations):
        assert n_iterations == 1
        J = []
        df = 1
        Jep = 0
        self._init_update()
        for sample in dataset:
            x, u, r, xn, _, last = self._parse(sample)
            Jep += df*r
            df *= self._gamma
            self._step_update(x,u)

            if last:
                self._episode_end_update(Jep)
                J.append(Jep)
                Jep = 0
                df = 1
                self._init_update()

        self._update_parameters(J)

    def _update_parameters(self, J):
        grad_J = self._compute_gradient(J)
        theta = self.policy.get_weights()
        theta_new = theta+self.learning_rate(grad_J)*grad_J
        self.policy.set_weights(theta_new)

    def _init_update(self):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _compute_gradient(self, J):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _step_update(self,x,u):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _episode_end_update(self, Jep):
        raise NotImplementedError('Policy gradient is an abstract class')

    def _parse(self, sample):
        x = sample[0]
        u = sample[1]
        r = sample[2]
        xn = sample[3]
        ab = sample[4]
        last = sample[5]

        if self.phi is not None:
            x = self.phi(x)

        return x, u, r, xn, ab, last

    def __str__(self):
        return self.__name__
