import numpy as np

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator


class StochasticAC(Agent):
    """
    Stochastic Actor critic in the episodic setting as presented in:
    "Model-Free Reinforcement Learning with Continuous Action in Practice".
    Degris T. et al.. 2012.

    """
    def __init__(self, mdp_info, policy, alpha_theta, alpha_v, lambda_par=.9,
                 value_function_features=None, policy_features=None):
        """
        Constructor.

        Args:
            alpha_theta (Parameter): learning rate for policy update;
            alpha_v (Parameter): learning rate for the value function;
            lambda_par (float, .9): trace decay parameter;
            value_function_features (Features, None): features used by the
                value function approximator;
            policy_features (Features, None): features used by the policy.

        """
        self._psi = value_function_features

        self._alpha_theta = alpha_theta
        self._alpha_v = alpha_v

        self._lambda = lambda_par

        super().__init__(mdp_info, policy, policy_features)

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = Regressor(LinearApproximator, input_shape=input_shape,
                            output_shape=(1,))

        self._e_v = np.zeros(self._V.weights_size)
        self._e_theta = np.zeros(self.policy.weights_size)

        self._add_save_attr(
            _psi='pickle',
            _alpha_theta='pickle',
            _alpha_v='pickle',
            _lambda='primitive',
            _V='mushroom',
            _e_v='numpy',
            _e_theta='numpy'
        )

    def episode_start(self):
        self._e_v = np.zeros(self._V.weights_size)
        self._e_theta = np.zeros(self.policy.weights_size)

        super().episode_start()

    def fit(self, dataset):
        for step in dataset:
            s, a, r, ss, absorbing, _ = step

            s_phi = self.phi(s) if self.phi is not None else s
            s_psi = self._psi(s) if self._psi is not None else s
            ss_psi = self._psi(ss) if self._psi is not None else ss

            v_next = self._V(ss_psi) if not absorbing else 0

            delta = self._compute_td_n_traces(a, r, v_next, s_psi, s_phi)

            # Update value function
            delta_v = self._alpha_v(s, a) * delta * self._e_v
            v_new = self._V.get_weights() + delta_v
            self._V.set_weights(v_new)

            # Update policy
            delta_theta = self._alpha_theta(s, a) * delta * self._e_theta
            theta_new = self.policy.get_weights() + delta_theta
            self.policy.set_weights(theta_new)

    def _compute_td_n_traces(self, a, r, v_next, s_psi, s_phi):
        # Compute TD error
        delta = r + self.mdp_info.gamma * v_next - self._V(s_psi)

        # Update traces
        self._e_v = self.mdp_info.gamma * self._lambda * self._e_v + s_psi
        self._e_theta = self.mdp_info.gamma * self._lambda * \
            self._e_theta + self.policy.diff_log(s_phi, a)

        return delta


class StochasticAC_AVG(StochasticAC):
    """
    Stochastic Actor critic in the average reward setting as presented in:
    "Model-Free Reinforcement Learning with Continuous Action in Practice".
    Degris T. et al.. 2012.

    """
    def __init__(self, mdp_info, policy, alpha_theta, alpha_v, alpha_r,
                 lambda_par=.9, value_function_features=None,
                 policy_features=None):
        """
        Constructor.

        Args:
            alpha_r (Parameter): learning rate for the reward trace.

        """
        super().__init__(mdp_info, policy, alpha_theta, alpha_v, lambda_par,
                         value_function_features, policy_features)

        self._alpha_r = alpha_r
        self._r_bar = 0

        self._add_save_attr(_alpha_r='pickle', _r_bar='primitive')

    def _compute_td_n_traces(self, a, r, v_next, s_psi, s_phi):
        # Compute TD error
        delta = r - self._r_bar + v_next - self._V(s_psi)

        # Update traces
        self._r_bar += self._alpha_r() * delta
        self._e_v = self._lambda * self._e_v + s_psi
        self._e_theta = self._lambda * self._e_theta + \
                        self.policy.diff_log(s_phi, a)

        return delta
