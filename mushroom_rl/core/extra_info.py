from mushroom_rl.core.mushroom_object import MushroomObject

from ._impl.extra_info import StepInfo, EpisodeInfo


class ExtraInfo(MushroomObject):
    """
    Container for everything a dataset stores beside the transitions: the step information, the episode
    information and the per-episode policy parameters.

    """
    def __init__(self, n_envs, backend, device=None, vectorized=None, theta_backend=None, theta_device=None):
        """
        Constructor.

        Args:
            n_envs (int): number of parallel environments the information is reported for;
            backend (str): name of the array backend the parsed arrays are built in;
            device (str, None): device the parsed arrays are placed on;
            vectorized (bool, None): whether the information is provided in vectorized form. If None, it
                defaults to ``n_envs > 1``;
            theta_backend (str, None): name of the array backend of the policy parameters;
            theta_device (str, None): device the policy parameters are placed on.

        """
        self._n_envs = n_envs
        self._backend = backend
        self._device = device
        self._vectorized = n_envs > 1 if vectorized is None else vectorized
        self._theta_backend = theta_backend if theta_backend is not None else backend
        self._theta_device = theta_device

        self._step_info = StepInfo(n_envs, backend, device, vectorized=self._vectorized)
        self._episode_info = EpisodeInfo(n_envs, backend, device, vectorized=self._vectorized)
        self._theta = EpisodeInfo(n_envs, self._theta_backend, self._theta_device, vectorized=self._vectorized)

        self._add_save_attr(
            _n_envs='primitive',
            _backend='primitive',
            _device='primitive',
            _vectorized='primitive',
            _theta_backend='primitive',
            _theta_device='primitive',
            _step_info='mushroom',
            _episode_info='mushroom',
            _theta='mushroom'
        )

    def __add__(self, other):
        """
        Combine two ExtraInfo objects into a new one.

        Args:
            other (ExtraInfo): ExtraInfo which will be combined with this one.

        Returns:
            A new ExtraInfo holding the content of both.

        """
        extras = self.copy()
        extras += other

        return extras

    def __iadd__(self, other):
        """
        In-place append: extend every container with the content of another ExtraInfo.

        Args:
            other (ExtraInfo): ExtraInfo whose content will be appended.

        """
        self._step_info += other._step_info
        self._episode_info += other._episode_info
        self._theta += other._theta

        return self

    def append_step(self, info):
        """
        Append the step information of one step.

        Args:
            info (dict or list): the information the step reported.

        """
        self._step_info.append(info)

    def append_episode(self, info, mask=None):
        """
        Append the information reported by a reset, keeping only the environments that were reset.

        Args:
            info (dict or list): the information the reset reported;
            mask (Array, None): boolean mask selecting the environments that were reset.

        """
        self._episode_info.append(info, mask)

    def append_theta(self, theta):
        """
        Append the policy parameters of the episode that is starting on the only environment.

        Args:
            theta (Array): the policy parameters.

        """
        self._theta.append(theta)

    def append_theta_vectorized(self, theta, mask):
        """
        Append the policy parameters of the environments the mask selects.

        Args:
            theta (Array): the policy parameters, one entry per environment;
            mask (Array): boolean mask selecting the environments that were reset.

        """
        self._theta.append(theta, mask)

    def parse_steps(self):
        """
        Returns:
            A flat dictionary holding an array per key of the step information.

        """
        return self._step_info.parse()

    def parse_episodes(self):
        """
        Returns:
            A flat dictionary holding an array per key of the episode information.

        """
        return self._episode_info.parse()

    def to_backend(self, backend, device=None):
        """
        Copy the information, to be parsed in another backend.

        Args:
            backend (str): name of the array backend the parsed arrays are built in;
            device (str, None): device the parsed arrays are placed on, or ``None`` for the default one.

        Returns:
            A new ExtraInfo holding the same content, parsed in the given backend.

        """
        extras = ExtraInfo(self._n_envs, backend, device, vectorized=self._vectorized,
                           theta_backend=backend, theta_device=device)
        extras._step_info = self._step_info.to_backend(backend, device)
        extras._episode_info = self._episode_info.to_backend(backend, device)
        extras._theta = self._theta.to_backend(backend, device)

        return extras

    def get_view(self, index, copy=False):
        """
        Select a subset of the stored steps. The episode information and the policy parameters are dropped.

        Args:
            index (int, slice, ndarray, tensor): the steps the result should contain;
            copy (bool, False): whether the content should be copied rather than shared.

        Returns:
            A new ExtraInfo holding only the selected steps.

        """
        extras = self.empty()
        extras._step_info = self._step_info.get_view(index, copy)

        return extras

    def keep_from(self, first_step):
        """
        Drop the steps before the given one, along with the episode information and the policy parameters.

        Args:
            first_step (int): index of the first step to keep.

        """
        self._step_info.drop_before(first_step)
        self._episode_info.clear()
        self._theta.clear()

    def flatten(self, mask=None):
        """
        Combine the environment dimension of the step information into the step one, keeping the entries the
        mask selects, and concatenate the episodes of every environment.

        Args:
            mask (Array, None): boolean mask selecting, for every step, the environments to keep.

        Returns:
            A single-environment ExtraInfo.

        """
        extras = ExtraInfo(1, self._backend, self._device, vectorized=False,
                           theta_backend=self._theta_backend, theta_device=self._theta_device)
        extras._step_info = self._step_info.flatten(mask)
        extras._episode_info = self._episode_info.flatten()
        extras._theta = self._theta.flatten()

        return extras

    def reorder_steps(self, index):
        """
        Copy the information with the steps reordered.

        Args:
            index (Array): the step placed at every position.

        Returns:
            A copy with the steps in the given order and the same episode information and policy parameters.

        """
        extras = self.copy()
        extras._step_info = self._step_info.get_view(index, copy=True)

        return extras

    def empty(self):
        """
        Returns:
            A new ExtraInfo holding no content, shaped like this one.

        """
        return ExtraInfo(self._n_envs, self._backend, self._device, vectorized=self._vectorized,
                         theta_backend=self._theta_backend, theta_device=self._theta_device)

    def copy(self):
        """
        Returns:
            A new ExtraInfo holding the same content.

        """
        extras = self.empty()
        extras._step_info = self._step_info.copy()
        extras._episode_info = self._episode_info.copy()
        extras._theta = self._theta.copy()

        return extras

    def clear(self):
        """
        Drop all the stored content.

        """
        self._step_info.clear()
        self._episode_info.clear()
        self._theta.clear()

    @property
    def theta(self):
        """
        Returns:
            The policy parameters, one list of per-episode entries per environment, or a single list of
            per-episode entries when they are collected from one environment.

        """
        return self._theta.episodes
