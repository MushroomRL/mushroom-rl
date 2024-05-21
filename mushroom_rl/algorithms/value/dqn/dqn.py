from mushroom_rl.algorithms.value.dqn import AbstractDQN


class DQN(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        if absorbing.any():
            q *= 1 - absorbing.reshape(-1, 1)

        return q.max(1)
