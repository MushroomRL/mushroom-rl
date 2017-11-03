from mushroom.algorithms.agent import Agent


class PolicyGradient(Agent):
    """
    Abstract class to implement a generic Policy Search algorithm using the
    gradient of the policy to update its parameters.
    "A survey on Policy Gradient algorithms for Robotics". Deisenroth M. P. et
    al.. 2011.

    """
    def __init__(self, policy, gamma, params, features):
        """
        Constructor.

        Args:
             policy (object): a differentiable policy;
             gamma (float): the discount factor;
             params (dict): other parameters;
             features (object): the features to use to preprocess the
                state.

        """
        self.learning_rate = params['algorithm_params'].pop('learning_rate')
        self.J_episode = None
        self.df = None

        super(PolicyGradient, self).__init__(policy, gamma, params, features)

    def fit(self, dataset, n_iterations):
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
                J_episode = 0.
                df = 1.
                self._init_update()

        self._update_parameters(J)

    def _update_parameters(self, J):
        """
        Update the parameters of the policy.

        Args:
             J (list): list of the cumulative discounted rewards for each
                episode in the dataset.

        """
        grad_J = self._compute_gradient(J)
        theta = self.policy.get_weights()
        theta_new = theta + self.learning_rate(grad_J) * grad_J
        self.policy.set_weights(theta_new)

    def _init_update(self):
        """
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE reset some data structure).

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _step_update(self, state, action):
        """
        This function is called, when parsing the dataset, at each episode step.

        Args:
            state (np.array): the state at the current step;
            action (np.array): the action at the current step.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _episode_end_update(self, J_episode):
        """
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE updates some data structures).

        Args:
            J_episode (float): cumulative discounted reward of the current
                episode.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _compute_gradient(self, J):
        """
        Return the gradient computed by the algorithm.

        Args:
             J (list): list of the cumulative discounted rewards for each
                episode in the dataset.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _parse(self, sample):
        """
        Utility to parse the sample.

        Args:
             sample (list): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag. If provided, `state` is preprocessed with the features.

        """
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
