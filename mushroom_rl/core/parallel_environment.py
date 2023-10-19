from .environment import Environment


class ParallelEnvironment(object):
    """
    Basic interface to generate and collect multiple copies of the same environment.
    This class assumes that the environments are homogeneus, i.e. have the same type and MDP info.

    """
    def __init__(self, env_list):
        """
        Constructor.

        Args:
            env_list: list of the environments to be evaluated in parallel.

        """
        self.envs = env_list

    @property
    def info(self):
        """
        Returns:
             An object containing the info of all environments.

        """
        return self.envs[0].info

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, item):
        return self.envs[item]

    def seed(self, seeds):
        """
        Set the seed of all environments.

        Args:
            seeds ([int, list]): the value of the seed or a list of seeds for each environment. The list lenght must be
                equal to the number of parallel environments.

        """
        if isinstance(seeds, list):
            assert len(seeds) == len(self)
            for env, seed in zip(self.envs,seeds):
                env.seed(seed)
        else:
            for env in self.envs:
                env.seed(seeds)

    def stop(self):
        """
        Method used to stop an mdp. Useful when dealing with real world environments, simulators, or when using
        openai-gym rendering

        """
        for env in self.envs:
            env.stop()

    @staticmethod
    def make(env_name, n_envs, use_constructor=False, *args, **kwargs):
        """
        Generate multiple copies of a given environment using the specified name and parameters.
        The environment is created using the generate method, if available. Otherwise, the constructor is used.
        See the `Environment.make` documentation for more information.

        Args:
            env_name (str): Name of the environment;
            n_envs (int): Number of environments in parallel to generate;
            use_constructor (bool, False): whether to force the method to use the constructor instead of the generate
                method;
            *args: positional arguments to be provided to the environment generator/constructor;
            **kwargs: keyword arguments to be provided to the environment generator/constructor.

        Returns:
            An instance of the constructed environment.

        """
        if '.' in env_name:
            env_data = env_name.split('.')
            env_name = env_data[0]
            args = env_data[1:] + list(args)

        env = Environment._registered_envs[env_name]

        if not use_constructor and hasattr(env, 'generate'):
            return ParallelEnvironment.generate(env, *args, **kwargs)
        else:
            return ParallelEnvironment([env(*args, **kwargs) for _ in range(n_envs)])

    @staticmethod
    def init(env, n_envs, *args, **kwargs):
        """
        Method to generate an array of multiple copies of the same environment, calling the constructor n_envs times

        Args:
            env (class): the environment to be constructed;
            *args: positional arguments to be passed to the constructor;
            n_envs (int, 1): number of environments to generate;
            **kwargs: keywords arguments to be passed to the constructor

        Returns:
            A list containing multiple copies of the environment.

        """
        return

    @staticmethod
    def generate(env, n_envs, *args, **kwargs):
        """
        Method to generate an array of multiple copies of the same environment, calling the generate method n_envs times

        Args:
            env (class): the environment to be constructed;
            *args: positional arguments to be passed to the constructor;
            n_envs (int, 1): number of environments to generate;
            **kwargs: keywords arguments to be passed to the constructor

        Returns:
            A list containing multiple copies of the environment.

        """
        return ParallelEnvironment([env.generate(*args, **kwargs) for _ in range(n_envs)])