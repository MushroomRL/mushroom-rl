import torch

from mushroom_rl.algorithms.value.dqn import DoubleDQN


class DSORDQN(DoubleDQN):
    """
    Double Successive Over-Relaxation Deep Q-Network algorithm.
    "Double Successive Over-Relaxation Q-Learning with an Extension to Deep
    Reinforcement Learning".
    Shreyas S. R. 2025.

    """
    def __init__(self, mdp_info, policy, approximator, relaxation_factor,
                 **params):
        """
        Constructor.

        Args:
            relaxation_factor (float): the SOR relaxation factor. A value of
                one recovers Double DQN.

        """
        if relaxation_factor <= 0.:
            raise ValueError('The relaxation factor must be positive.')

        self._relaxation_factor = relaxation_factor

        super().__init__(mdp_info, policy, approximator, **params)

        self._add_save_attr(
            _relaxation_factor='primitive'
        )

    def _compute_target(self, state, reward, next_state, absorbing):
        q_next = self._double_q(next_state)
        q_current = self._double_q(state)

        if absorbing.any():
            q_next *= ~absorbing

        w = self._relaxation_factor
        return w * (reward + self.mdp_info.gamma * q_next) + \
            (1. - w) * q_current

    def _double_q(self, state):
        q = self.approximator.predict(state, **self._predict_params)
        max_a = torch.argmax(q, 1).unsqueeze(1)

        return self.target_approximator.predict(
            state, max_a, **self._predict_params)
