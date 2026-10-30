from mushroom_rl.policy.policy import Policy, HasWeights


class DeterministicPolicy(HasWeights, Policy):
    """
    Simple parametric policy representing a deterministic policy. As
    deterministic policies are degenerate probability functions where all
    the probability mass is on the deterministic action,they are not
    differentiable, even if the mean value approximator is differentiable.

    """
    def __init__(self, mu):
        """
        Constructor.

        Args:
            mu (Approximator): the approximator representing the action to select
                in each state. The policy operates in its backend and on its device.

        """
        super().__init__(mu.backend_name, mu.device)

        self._approximator = mu
        self._predict_params = dict()

        self._add_save_attr(_approximator='mushroom',
                            _predict_params='pickle')

    def get_regressor(self):
        """
        Getter.

        Returns:
            The regressor that is used to map state to actions.

        """
        return self._approximator

    def __call__(self, state, action):
        policy_action = self._approximator.predict(state, **self._predict_params)

        return 1. if action.shape == policy_action.shape and (action == policy_action).all() else 0.

    def draw_action(self, state):
        return self.draw_action_greedy(state)

    def draw_action_greedy(self, state):
        return self._approximator(state, **self._predict_params)

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size
