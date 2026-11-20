import numpy as np

from mushroom_rl.algorithms.actor_critic.classic_actor_critic.classic_actor_critic import ClassicAC
from mushroom_rl.rl_utils.parameters import Parameter


class StochasticAC(ClassicAC):
    """
    Stochastic Actor critic in the episodic setting as presented in:
    "Model-Free Reinforcement Learning with Continuous Action in Practice"
    Degris T. et al. 2012.

    """

    def __init__(self, mdp_info, policy, alpha_theta, alpha_v, lambda_par=.9, value_function_features=None):
        """
        Constructor.

        Args:
            alpha_theta ([float, Parameter]): learning rate for policy update;
            alpha_v ([float, Parameter]): learning rate for the value function;
            lambda_par ([float, Parameter], .9): trace decay parameter;
            value_function_features (Features, None): features used by the value function approximator.

        """
        self._lambda = Parameter.make(lambda_par)

        super().__init__(mdp_info, policy, alpha_theta, alpha_v, value_function_features)

        self._e_v = np.zeros(self._V.weights_size)
        self._e_theta = np.zeros(self.policy.weights_size)

        self._add_save_attr(
            _lambda='mushroom',
            _e_v='numpy',
            _e_theta='numpy'
        )

    def episode_start(self, initial_state, episode_info, greedy=False):
        self._e_v = np.zeros(self._V.weights_size)
        self._e_theta = np.zeros(self.policy.weights_size)

        return super().episode_start(initial_state, episode_info, greedy)

    def episode_start_vectorized(self, initial_states, episode_info, start_mask, greedy=False):
        assert False, "StochasticAC does not support vectorized environments"

    def _update(self, state, action, reward, v_next, state_psi):
        delta = self._compute_td_n_traces(state, action, reward, v_next, state_psi)

        # Update value function
        delta_v = self._alpha_v(state, action) * delta * self._e_v
        v_new = self._V.get_weights() + delta_v
        self._V.set_weights(v_new)

        # Update policy
        delta_theta = self._alpha_theta(state, action) * delta * self._e_theta
        theta_new = self.policy.get_weights() + delta_theta
        self.policy.set_weights(theta_new)

    def _compute_td_n_traces(self, s, a, r, v_next, s_psi):
        # Compute TD error
        delta = r + self.mdp_info.gamma * v_next - self._V(s_psi)

        # Update traces
        self._e_v = self.mdp_info.gamma * self._lambda() * self._e_v + s_psi
        self._e_theta = self.mdp_info.gamma * self._lambda() * self._e_theta + self.policy.diff_log(s, a)

        return delta


class StochasticAC_AVG(StochasticAC):
    """
    Stochastic Actor critic in the average reward setting as presented in:
    "Model-Free Reinforcement Learning with Continuous Action in Practice".
    Degris T. et al.. 2012.

    """
    def __init__(self, mdp_info, policy, alpha_theta, alpha_v, alpha_r, lambda_par=.9, value_function_features=None):
        """
        Constructor.

        Args:
            alpha_r (Parameter): learning rate for the reward trace.

        """
        super().__init__(mdp_info, policy, alpha_theta, alpha_v, lambda_par,  value_function_features)

        self._alpha_r = Parameter.make(alpha_r)
        self._r_bar = 0

        self._add_save_attr(_alpha_r='mushroom', _r_bar='primitive')

    def _compute_td_n_traces(self, s, a, r, v_next, s_psi):
        # Compute TD error
        delta = r - self._r_bar + v_next - self._V(s_psi)

        # Update traces
        self._r_bar += self._alpha_r() * delta
        self._e_v = self._lambda() * self._e_v + s_psi
        self._e_theta = self._lambda() * self._e_theta + self.policy.diff_log(s, a)

        return delta
