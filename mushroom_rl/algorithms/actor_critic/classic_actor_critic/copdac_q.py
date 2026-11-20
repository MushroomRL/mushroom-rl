from mushroom_rl.algorithms.actor_critic.classic_actor_critic.classic_actor_critic import ClassicAC
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.rl_utils.parameters import Parameter


class COPDAC_Q(ClassicAC):
    """
    Compatible off-policy deterministic actor-critic algorithm.
    "Deterministic Policy Gradient Algorithms"
    Silver D. et al. 2014.

    The deterministic policy ``mu`` is saved separately from the behaviour policy: when the behaviour policy is built
    on ``mu``, a loaded agent holds two distinct copies of it, and only the agent's one is updated by the fit.

    """

    def __init__(self, mdp_info, policy, mu, alpha_theta, alpha_omega, alpha_v, value_function_features=None):
        """
        Constructor.

        Args:
            mu (Approximator): approximator that describe the deterministic policy to be
                learned i.e., the deterministic mapping between state and action.
            alpha_theta ([float, Parameter]): learning rate for policy update;
            alpha_omega ([float, Parameter]): learning rate for the advantage function;
            alpha_v ([float, Parameter]): learning rate for the value function;
            value_function_features (Features, None): features used by the value
                function approximator;

        """
        self._mu = mu
        self._alpha_omega = Parameter.make(alpha_omega)

        self._A = LinearApproximator(input_shape=(self._mu.weights_size,), output_shape=())

        super().__init__(mdp_info, policy, alpha_theta, alpha_v, value_function_features)

        self._add_save_attr(
            _mu='mushroom',
            _alpha_omega='mushroom',
            _A='mushroom'
        )
        self._add_logger_attr(_alpha_omega='omega', group='alpha')

    def _update(self, state, action, reward, v_next, state_psi):
        grad_mu_s = self._mu.diff(state)
        omega = self._A.get_weights()

        delta = reward + self.mdp_info.gamma * v_next - self._Q(state, action)
        delta_theta = self._alpha_theta(state, action) * (grad_mu_s @ (grad_mu_s.T @ omega))
        delta_omega = self._alpha_omega(state, action) * delta * self._nu(state, action)
        delta_v = self._alpha_v(state, action) * delta * state_psi

        theta_new = self._mu.get_weights() + delta_theta
        self._mu.set_weights(theta_new)

        omega_new = omega + delta_omega
        self._A.set_weights(omega_new)

        v_new = self._V.get_weights() + delta_v
        self._V.set_weights(v_new)

    def _Q(self, state, action):
        state_psi = self._psi(state) if self._psi is not None else state

        return self._V(state_psi) + self._A(self._nu(state, action))

    def _nu(self, state, action):
        grad_mu = self._mu.diff(state)
        delta = action - self._mu(state)

        return grad_mu @ delta
