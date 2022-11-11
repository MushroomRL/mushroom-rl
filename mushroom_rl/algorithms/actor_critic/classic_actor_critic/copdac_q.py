import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator

from mushroom_rl.utils.parameters import to_parameter


class COPDAC_Q(Agent):
    """
    Compatible off-policy deterministic actor-critic algorithm.
    "Deterministic Policy Gradient Algorithms".
    Silver D. et al.. 2014.

    """
    def __init__(self, mdp_info, policy, mu, alpha_theta, alpha_omega, alpha_v,
                 value_function_features=None, policy_features=None):
        """
        Constructor.

        Args:
            mu (Regressor): regressor that describe the deterministic policy to be
                learned i.e., the deterministic mapping between state and action.
            alpha_theta ([float, Parameter]): learning rate for policy update;
            alpha_omega ([float, Parameter]): learning rate for the advantage function;
            alpha_v ([float, Parameter]): learning rate for the value function;
            value_function_features (Features, None): features used by the value
                function approximator;
            policy_features (Features, None): features used by the policy.

        """
        self._mu = mu
        self._psi = value_function_features

        self._alpha_theta = to_parameter(alpha_theta)
        self._alpha_omega = to_parameter(alpha_omega)
        self._alpha_v = to_parameter(alpha_v)

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = Regressor(LinearApproximator, input_shape=input_shape,
                            output_shape=(1,))

        self._A = Regressor(LinearApproximator,
                            input_shape=(self._mu.weights_size,),
                            output_shape=(1,))

        self._add_save_attr(
            _mu='mushroom',
            _psi='pickle',
            _alpha_theta='mushroom',
            _alpha_omega='mushroom',
            _alpha_v='mushroom',
            _V='mushroom',
            _A='mushroom'
        )

        super().__init__(mdp_info, policy, policy_features)

    def fit(self, dataset, **info):
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
