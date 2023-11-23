import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


class LQR(Environment):
    """
    This class implements a Linear-Quadratic Regulator.
    This task aims to minimize the undesired deviations from nominal values of
    some controller settings in control problems.
    The system equations in this task are:

    .. math::
        x_{t+1} = Ax_t + Bu_t

    where x is the state and u is the control signal.

    The reward function is given by:

    .. math::
        r_t = -\\left( x_t^TQx_t + u_t^TRu_t \\right)

    "Policy gradient approaches for multi-objective sequential decision making".
    Parisi S., Pirotta M., Smacchia N., Bascetta L., Restelli M.. 2014

    """
    def __init__(self, A, B, Q, R, max_pos=np.inf, max_action=np.inf,  random_init=False, episodic=False, gamma=0.9,
                 horizon=50, initial_state=None, dt=0.1):
        """
        Constructor.

            Args:
                A (np.ndarray): the state dynamics matrix;
                B (np.ndarray): the action dynamics matrix;
                Q (np.ndarray): reward weight matrix for state;
                R (np.ndarray): reward weight matrix for action;
                max_pos (float, np.inf): maximum value of the state;
                max_action (float, np.inf): maximum value of the action;
                random_init (bool, False): start from a random state;
                episodic (bool, False): end the episode when the state goes over
                the threshold;
                gamma (float, 0.9): discount factor;
                horizon (int, 50): horizon of the env;
                dt (float, 0.1): the control timestep of the environment.

        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self._max_pos = max_pos
        self._max_action = max_action
        self._episodic = episodic
        self.random_init = random_init

        self._initial_state = initial_state

        # MDP properties
        high_x = self._max_pos * np.ones(A.shape[0])
        low_x = -high_x

        high_u = self._max_action * np.ones(B.shape[1])
        low_u = -high_u

        observation_space = spaces.Box(low=low_x, high=high_x)
        action_space = spaces.Box(low=low_u, high=high_u)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        self._state = None

        super().__init__(mdp_info)

    @staticmethod
    def generate(dimensions=None, s_dim=None, a_dim=None, max_pos=np.inf, max_action=np.inf, eps=.1,
                 index=0, scale=1.0, random_init=False, episodic=False,
                 gamma=.9, horizon=50, initial_state=None):
        """
        Factory method that generates an lqr with identity dynamics and
        symmetric reward matrices.

        Args:
            dimensions (int): number of state-action dimensions;
            s_dim (int): number of state dimensions;
            a_dim (int): number of action dimensions;
            max_pos (float, np.inf): maximum value of the state;
            max_action (float, np.inf): maximum value of the action;
            eps (double, .1): reward matrix weights specifier;
            index (int, 0): selector for the principal state;
            scale (float, 1.0): scaling factor for the reward function;
            random_init (bool, False): start from a random state;
            episodic (bool, False): end the episode when the state goes over the
                threshold;
            gamma (float, .9): discount factor;
            horizon (int, 50): horizon of the env.

        """
        assert dimensions != None or (s_dim != None and a_dim != None)

        if s_dim == None or a_dim == None:
            s_dim = dimensions
            a_dim = dimensions
        A = np.eye(s_dim)
        B = np.eye(s_dim, a_dim)
        Q = eps * np.eye(s_dim) * scale
        R = (1. - eps) * np.eye(a_dim) * scale

        Q[index, index] = (1. - eps) * scale
        R[index, index] = eps * scale

        return LQR(A, B, Q, R, max_pos, max_action, random_init, episodic,
                   gamma, horizon, initial_state)

    def reset(self, state=None):
        if state is None:
            if self.random_init:
                rand_state = np.random.uniform(-3, 3, size=self.A.shape[0])
                self._state = self._bound(rand_state, self.info.observation_space.low, self.info.observation_space.high)
            elif self._initial_state is not None:
                self._state = self._initial_state
            else:
                init_value = .9 * self._max_pos if np.isfinite(self._max_pos) else 10
                self._state = init_value * np.ones(self.A.shape[0])
        else:
            self._state = state

        return self._state, {}

    def step(self, action):
        x = self._state
        u = self._bound(action, self.info.action_space.low, self.info.action_space.high)

        reward = -(x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))
        self._state = self.A.dot(x) + self.B.dot(u)

        absorbing = False

        if np.any(np.abs(self._state) > self._max_pos):
            if self._episodic:
                reward = -self._max_pos ** 2 * 10
                absorbing = True
            else:
                self._state = self._bound(self._state, self.info.observation_space.low,
                                          self.info.observation_space.high)

        return self._state, reward, absorbing, {}
