from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import TorchApproximator


class DQN(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al. 2015.

    """
    def __init__(self, mdp_info, policy, approximator=TorchApproximator, **params):
        """
        Constructor.

        Args:
            approximator (class, TorchApproximator): the approximator to use to fit the Q-function;
            **params: parameters of the base class.

        """
        super().__init__(mdp_info, policy, approximator, **params)

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        if absorbing.any():
            q *= ~absorbing.unsqueeze(1)

        return q.max(1).values
