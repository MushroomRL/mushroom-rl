import warnings
import numpy as np

from mushroom_rl.core.serialization import Serializable


class MDPInfo(Serializable):
    """
    This class is used to store the information of the environment.

    """
    def __init__(self, observation_space, action_space, gamma, horizon):
        """
        Constructor.

        Args:
             observation_space ([Box, Discrete]): the state space;
             action_space ([Box, Discrete]): the action space;
             gamma (float): the discount factor;
             horizon (int): the horizon.

        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

        self._add_save_attr(
            observation_space='mushroom',
            action_space='mushroom',
            gamma='primitive',
            horizon='primitive'
        )

    @property
    def size(self):
        """
        Returns:
            The sum of the number of discrete states and discrete actions. Only
            works for discrete spaces.

        """
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        """
        Returns:
            The concatenation of the shape tuple of the state and action
            spaces.

        """
        return self.observation_space.shape + self.action_space.shape


class Environment(object):
    """
    Basic interface used by any mushroom environment.

    """

    @classmethod
    def register(cls):
        """
        Register an environment in the environment list.

        """
        env_name = cls.__name__

        if env_name not in Environment._registered_envs:
            Environment._registered_envs[env_name] = cls

    @staticmethod
    def list_registered():
        """
        List registered environments.

        Returns:
             The list of the registered environments.

        """
        return list(Environment._registered_envs.keys())

    @staticmethod
    def make(env_name, *args, **kwargs):
        """
        Generate an environment given an environment name and parameters.
        The environment is created using the generate method, if available. Otherwise, the constructor is used.
        The generate method has a simpler interface than the constructor, making it easier to generate
        a standard version of the environment. If the environment name contains a '.' separator, the string
        is splitted, the first element is used to select the environment and the other elements are passed as
        positional parameters.

        Args:
            env_name (str): Name of the environment,
            *args: positional arguments to be provided to the environment generator;
            **kwargs: keyword arguments to be provided to the environment generator.

        Returns:
            An instance of the constructed environment.

        """

        if '.' in env_name:
            env_data = env_name.split('.')
            env_name = env_data[0]
            args = env_data[1:] + list(args)

        env = Environment._registered_envs[env_name]

        if hasattr(env, 'generate'):
            return env.generate(*args, **kwargs)
        else:
            return env(*args, **kwargs)

    def __init__(self, mdp_info):
        """
        Constructor.

        Args:
             mdp_info (MDPInfo): an object containing the info of the
                environment.

        """
        self._mdp_info = mdp_info

    def seed(self, seed):
        """
        Set the seed of the environment.

        Args:
            seed (float): the value of the seed.

        """
        if hasattr(self, 'env') and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        else:
            warnings.warn('This environment has no custom seed. '
                          'The call will have no effect. '
                          'You can set the seed manually by setting numpy/torch seed')

    def reset(self, state=None):
        """
        Reset the current state.

        Args:
            state (np.ndarray, None): the state to set to the current state.

        Returns:
            The current state.

        """
        raise NotImplementedError

    def step(self, action):
        """
        Move the agent from its current state according to the action.

        Args:
            action (np.ndarray): the action to execute.

        Returns:
            The state reached by the agent executing ``action`` in its current
            state, the reward obtained in the transition and a flag to signal
            if the next state is absorbing. Also, an additional dictionary is
            returned (possibly empty).

        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def stop(self):
        """
        Method used to stop an mdp. Useful when dealing with real world
        environments, simulators, or when using openai-gym rendering

        """
        pass

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info

    @staticmethod
    def _bound(x, min_value, max_value):
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))

    _registered_envs = dict()
