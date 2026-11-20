from mushroom_rl.core import Agent
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.rl_utils.parameters import Parameter


class ClassicAC(Agent):
    """
    Base class for classical actor-critic algorithms, learning a linear state-value critic from every transition.

    """
    def __init__(self, mdp_info, policy, alpha_theta, alpha_v, value_function_features=None):
        """
        Constructor.

        Args:
            alpha_theta ([float, Parameter]): learning rate for policy update;
            alpha_v ([float, Parameter]): learning rate for the value function;
            value_function_features (Features, None): features used by the value function approximator.

        """
        self._psi = value_function_features

        self._alpha_theta = Parameter.make(alpha_theta)
        self._alpha_v = Parameter.make(alpha_v)

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = LinearApproximator(input_shape=input_shape, output_shape=())

        super().__init__(mdp_info, policy)

        self._add_save_attr(
            _psi='pickle',
            _alpha_theta='mushroom',
            _alpha_v='mushroom',
            _V='mushroom'
        )
        self._add_logger_attr(_alpha_theta='theta', _alpha_v='v', group='alpha')

    def fit(self, dataset):
        for step in dataset:
            s, a, r, ss, absorbing, _ = step

            s_psi = self._psi(s) if self._psi is not None else s
            ss_psi = self._psi(ss) if self._psi is not None else ss

            v_next = self._V(ss_psi) if not absorbing else 0

            self._update(s, a, r, v_next, s_psi)

        if self._logger:
            self._logger.advance_step()

    def _update(self, state, action, reward, v_next, state_psi):
        """
        Update the actor and the critic with a transition.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray): the action;
            reward (float): the reward;
            v_next (float): the value of the next state, zero when it is absorbing;
            state_psi (np.ndarray): the value function features of the state.

        """
        raise NotImplementedError('ClassicAC is an abstract class')
