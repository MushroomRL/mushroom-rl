import numpy as np

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.core.array_backend import ArrayBackend


class DatasetInfo(MushroomObject):
    """
    Static information needed to build a :class:`Dataset`. A dataset keeps its data in two backend-aware groups:
    the environment data (state, action, reward, next state, absorbing and last flags) and the agent data (the
    policy state). This class stores the array backend and device of each group, together with the shapes and
    dtypes of the states and actions, the horizon, the discount factor and the number of parallel environments.
    Build it with the :meth:`create_dataset_info` (on-policy collection) or :meth:`create_replay_memory_info`
    (replay buffer) factories.

    """
    def __init__(self, env_backend, agent_backend, env_device, agent_device, horizon, gamma, state_shape, state_dtype,
                 action_shape, action_dtype, policy_state_shape, n_envs=1):
        """
        Constructor.

        Args:
            env_backend (str): array backend of the environment data (``'numpy'``, ``'torch'`` or ``'list'``);
            agent_backend (str): array backend of the agent (policy state) data;
            env_device (str, None): device of the environment data, only allowed with the torch backend;
            agent_device (str, None): device of the agent data, only allowed with the torch backend;
            horizon (int): horizon of the MDP;
            gamma (float): discount factor;
            state_shape (tuple): shape of a single state;
            state_dtype: data type of the states;
            action_shape (tuple): shape of a single action;
            action_dtype: data type of the actions;
            policy_state_shape (tuple, None): shape of the policy state, or ``None`` if the agent is stateless;
            n_envs (int, 1): number of parallel environments.

        """
        assert env_backend == "torch" or env_device is None
        assert agent_backend == "torch" or agent_device is None

        self.env_backend = env_backend
        self.agent_backend = agent_backend
        self.env_device = env_device
        self.agent_device = agent_device
        self.horizon = horizon
        self.gamma = gamma
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.policy_state_shape = policy_state_shape
        self.n_envs = n_envs

        self._add_save_attr(
            env_backend='primitive',
            agent_backend='primitive',
            env_device='primitive',
            agent_device='primitive',
            gamma='primitive',
            horizon='primitive',
            state_shape='primitive',
            state_dtype='primitive',
            action_shape='primitive',
            action_dtype='primitive',
            policy_state_shape='primitive',
            n_envs='primitive'
        )

    def flat(self):
        """
        Returns:
            A copy of this dataset info describing a single, non-vectorized environment.

        """
        return DatasetInfo(self.env_backend, self.agent_backend, self.env_device, self.agent_device, self.horizon,
                           self.gamma, self.state_shape, self.state_dtype, self.action_shape, self.action_dtype,
                           self.policy_state_shape)

    @staticmethod
    def create_dataset_info(mdp_info, agent_info, n_envs=1):
        """
        Build the dataset info for on-policy collection: the environment data uses ``mdp_info.backend`` (forced
        to ``'list'`` for infinite-horizon MDPs) and the agent data uses ``agent_info.backend``.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            n_envs (int, 1): number of parallel environments.

        Returns:
            The dataset info.

        """
        env_backend = mdp_info.backend
        if not np.isfinite(mdp_info.horizon):
            assert env_backend != 'torch', "Infinite-horizon collection is not supported for the torch backend."
            assert agent_info.policy_state_shape is None or agent_info.backend != 'torch', \
                "Infinite-horizon collection is not supported for a stateful torch agent."
            env_backend = 'list'
        env_device = mdp_info.device
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = mdp_info.observation_space.data_type
        action_shape = mdp_info.action_space.shape
        action_dtype = mdp_info.action_space.data_type
        policy_state_shape = agent_info.policy_state_shape
        agent_device = agent_info.device

        return DatasetInfo(env_backend, agent_info.backend, env_device, agent_device, horizon, gamma,
                           state_shape, state_dtype, action_shape, action_dtype, policy_state_shape, n_envs=n_envs)

    @staticmethod
    def create_replay_memory_info(mdp_info, agent_info, store_policy_state=True):
        """
        Build the dataset info for a replay memory: the whole buffer (both the transition data and the policy
        state) lives in the agent backend, so the environment and agent backends/devices coincide.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            store_policy_state (bool, True): whether the policy state is stored.

        Returns:
            The dataset info.

        """
        backend = agent_info.backend
        array_backend = ArrayBackend.get_array_backend(backend)
        device = agent_info.device
        horizon = mdp_info.horizon
        gamma = mdp_info.gamma
        state_shape = mdp_info.observation_space.shape
        state_dtype = array_backend.to_backend_dtype(mdp_info.observation_space.data_type)
        action_shape = mdp_info.action_space.shape
        action_dtype = array_backend.to_backend_dtype(mdp_info.action_space.data_type)
        policy_state_shape = agent_info.policy_state_shape if store_policy_state else None

        return DatasetInfo(backend, backend, device, device, horizon, gamma, state_shape, state_dtype,
                           action_shape, action_dtype, policy_state_shape)

    @property
    def env_array_backend(self):
        """
        The :class:`ArrayBackend` of the environment data.

        """
        return ArrayBackend.get_array_backend(self.env_backend)

    @property
    def agent_array_backend(self):
        """
        The :class:`ArrayBackend` of the agent (policy state) data.

        """
        return ArrayBackend.get_array_backend(self.agent_backend)
