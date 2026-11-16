from mushroom_rl.policy.policy import Policy, HasMeanApproximator


class DeterministicPolicy(HasMeanApproximator, Policy):
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
        super().__init__(mu, backend=mu.backend_name, device=mu.device)

    def get_regressor(self):
        """
        Getter.

        Returns:
            The approximator that is used to map state to actions.

        """
        return self._approximator

    def __call__(self, state, action):
        policy_action = self._approximator.predict(state, **self._predict_params)

        return 1. if action.shape == policy_action.shape and (action == policy_action).all() else 0.

    def draw_action(self, state):
        return self.draw_action_greedy(state)

    def draw_action_greedy(self, state):
        return self.location(state)
