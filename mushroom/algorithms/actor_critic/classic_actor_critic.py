import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator


class StochasticAC(Agent):
    """
    Stochastic Actor critic in the episodic setting as presented in:
    "Model-Free Reinforcement Learning with Continuous Action in Practice".
    Degris T. et al.. 2012.

    """
    def __init__(self, policy, mdp_info, alpha_theta, alpha_v, lambda_par=.9,
                 value_function_features=None, policy_features=None):
        """
        Constructor.

        Args:
            policy (ParametricPolicy): a differentiable stochastic policy;
            mdp_info: information about the MDP;
            alpha_theta (Parameter): learning rate for policy update;
            alpha_v (Parameter): learning rate for the value function;
            lambda_par (float, 0.9): trace decay parameter;
            value_function_features (Features, None): features used by the value
                function approximator;
            policy_features (Features, None): features used by the policy.

        """
        self._psi = value_function_features

        self._alpha_theta = alpha_theta
        self._alpha_v = alpha_v

        self._lambda = lambda_par

        super().__init__(policy, mdp_info, policy_features)

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = Regressor(LinearApproximator, input_shape=input_shape,
                            output_shape=(1,))

        self._e_v = np.zeros(self._V.weights_size)
        self._e_theta = np.zeros(self.policy.weights_size)

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

            # Compute TD error
            delta = r + self.mdp_info.gamma * v_next - self._V(s_psi)

            # Update traces
            self._e_v = self.mdp_info.gamma * self._lambda * self._e_v + s_psi
            self._e_theta = self.mdp_info.gamma * self._lambda * \
                self._e_theta + self.policy.diff_log(s_phi, a)

            # Update value function
            delta_v = self._alpha_v(s, a) * delta * self._e_v
            v_new = self._V.get_weights() + delta_v
            self._V.set_weights(v_new)

            # Update policy
            delta_theta = self._alpha_theta(s, a) * delta * self._e_theta
            theta_new = self.policy.get_weights() + delta_theta
            self.policy.set_weights(theta_new)


class StochasticAC_AVG(Agent):
    """
    Stochastic Actor critic in the average reward setting as presented in:
    "Model-Free Reinforcement Learning with Continuous Action in Practice".
    Degris T. et al.. 2012.

    """
    def __init__(self, policy, mdp_info, alpha_theta, alpha_v, alpha_r,
                 lambda_par=.9, value_function_features=None,
                 policy_features=None):
        """
        Constructor.

        Args:
            policy (ParametricPolicy): a differentiable stochastic policy;
            mdp_info: information about the MDP;
            alpha_theta (Parameter): learning rate for policy update;
            alpha_v (Parameter): learning rate for the value function;
            alpha_r (Parameter): learning rate for the reward trace;
            lambda_par (float, 0.9): trace decay parameter;
            value_function_features (Features, None): features used by the
                value function approximator;
            policy_features (Features, None): features used by the policy.

        """
        self._psi = value_function_features

        self._alpha_theta = alpha_theta
        self._alpha_v = alpha_v
        self._alpha_r = alpha_r

        self._lambda = lambda_par

        super().__init__(policy, mdp_info, policy_features)

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = Regressor(LinearApproximator, input_shape=input_shape,
                            output_shape=(1,))

        self._e_v = np.zeros(self._V.weights_size)
        self._e_theta = np.zeros(self.policy.weights_size)
        self._r_bar = 0

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

            # Compute TD error
            delta = r - self._r_bar + v_next - self._V(s_psi)

            # Update traces
            self._r_bar += self._alpha_r() * delta
            self._e_v = self._lambda * self._e_v + s_psi
            self._e_theta = self._lambda * self._e_theta + \
                self.policy.diff_log(s_phi, a)

            # Update value function
            delta_v = self._alpha_v(s, a) * delta * self._e_v
            v_new = self._V.get_weights() + delta_v
            self._V.set_weights(v_new)

            # Update policy
            delta_theta = self._alpha_theta(s, a) * delta * self._e_theta
            theta_new = self.policy.get_weights() + delta_theta
            self.policy.set_weights(theta_new)


class COPDAC_Q(Agent):
    """
    Compatible off-policy deterministic actor-critic algorithm.
    "Deterministic Policy Gradient Algorithms".
    Silver D. et al.. 2014.

    """

    def __init__(self, policy, mu, mdp_info, alpha_theta, alpha_omega, alpha_v,
                 value_function_features=None, policy_features=None):
        """
        Constructor.

        Args:
            policy (Policy): any exploration policy, possibly using the deterministic
                policy as mean regressor;
            mu (Regressor): regressor that describe the deterministic policy to be
                learned i.e., the deterministic mapping between state and action.
            alpha_theta (Parameter): learning rate for policy update;
            alpha_omega (Parameter): learning rate for the advantage function;
            alpha_v (Parameter): learning rate for the value function;
            value_function_features (Features, None): features used by the value
                function approximator;
            policy_features (Features, None): features used by the policy.

        """
        self._mu = mu
        self._psi = value_function_features

        self._alpha_theta = alpha_theta
        self._alpha_omega = alpha_omega
        self._alpha_v = alpha_v

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = Regressor(LinearApproximator, input_shape=input_shape,
                            output_shape=(1,))

        self._A = Regressor(LinearApproximator,
                            input_shape=(self._mu.weights_size,),
                            output_shape=(1,))

        super().__init__(policy, mdp_info, policy_features)

    def fit(self, dataset):
        for step in dataset:
            s, a, r, ss, absorbing, _ = step

            s_phi = self.phi(s) if self.phi is not None else s
            s_psi = self._psi(s) if self._psi is not None else s
            ss_psi = self._psi(ss) if self._psi is not None else ss

            q_next = self._V(ss_psi).item() if not absorbing else 0

            grad_mu_s = np.atleast_2d(self._mu.diff(s_phi))
            omega = self._A.get_weights()

            delta = r + self.mdp_info.gamma * q_next - self._Q(s, a)
            delta_theta = self._alpha_theta(s, a) * \
                          omega.dot(grad_mu_s.T).dot(grad_mu_s)
            delta_omega = self._alpha_omega(s, a) * delta * self._nu(s, a)
            delta_v = self._alpha_v(s, a) * delta * s_psi

            theta_new = self._mu.get_weights() + delta_theta
            self._mu.set_weights(theta_new)

            omega_new = omega + delta_omega
            self._A.set_weights(omega_new)

            v_new = self._V.get_weights() + delta_v
            self._V.set_weights(v_new)

    def _Q(self, state, action):
        state_psi = self._psi(state) if self._psi is not None else state

        return self._V(state_psi).item() + self._A(self._nu(state,
                                                            action)).item()

    def _nu(self, state, action):
        state_phi = self.phi(state) if self.phi is not None else state
        grad_mu = np.atleast_2d(self._mu.diff(state_phi))
        delta = action - self._mu(state_phi)

        return delta.dot(grad_mu)
