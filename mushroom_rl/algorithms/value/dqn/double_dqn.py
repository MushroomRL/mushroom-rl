import torch

from mushroom_rl.algorithms.value.dqn import DQN


class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al. 2016.

    """
    def _next_q(self, next_state, absorbing):
        q = self.approximator.predict(next_state, **self._predict_params)
        max_a = torch.argmax(q, 1).unsqueeze(1)

        double_q = self.target_approximator.predict(next_state, max_a, **self._predict_params)
        if absorbing.any():
            double_q *= ~absorbing

        return double_q
