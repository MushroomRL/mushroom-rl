from collections import namedtuple
from enum import Enum

import torch
import warp as wp

from mushroom_rl.utils import TorchUtils

_EnumInfo = namedtuple('_EnumInfo', 'category length accessor component keyword', defaults=(None, None))


class ObservationType(Enum):
    """
    Enumeration for different types of observations in a simulation.

    Each observation type is described by an ``_EnumInfo`` carrying:

    - A category (`category`), either 'body', 'joint' or 'sub_body'.
    - The length of the observation (`length`).
    - The Isaac Sim call serving it (`accessor`), without the ``get_``/``set_`` prefix.
    - For a type that is one component of a call returning several of them, which component it is on the
      read side (`component`) and the keyword the setter takes it under on the write side (`keyword`).

    A type whose value is assembled from every component of such a call declares neither, and
    ``ObservationHelper`` combines them explicitly.
    """

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)  # Automatically assign ID based on current number of members
        return obj

    BODY_POS = _EnumInfo('body', 3, 'world_poses', component=0)
    BODY_ROT = _EnumInfo('body', 4, 'world_poses', component=1, keyword='orientations')
    BODY_LIN_VEL = _EnumInfo('body', 3, 'velocities', component=0, keyword='linear_velocities')
    BODY_ANG_VEL = _EnumInfo('body', 3, 'velocities', component=1, keyword='angular_velocities')
    BODY_VEL = _EnumInfo('body', 6, 'velocities')  # combination of lin_vel and ang_vel
    BODY_SCALE = _EnumInfo('body', 3, 'local_scales')
    JOINT_POS = _EnumInfo('joint', 1, 'dof_positions')
    JOINT_VEL = _EnumInfo('joint', 1, 'dof_velocities')
    JOINT_GAIN = _EnumInfo('joint', 2, 'dof_gains')  # combination of stiffness and damping
    JOINT_GAIN_STIFFNESS = _EnumInfo('joint', 1, 'dof_gains', component=0, keyword='stiffnesses')
    JOINT_GAIN_DAMPING = _EnumInfo('joint', 1, 'dof_gains', component=1, keyword='dampings')
    JOINT_DEFAULT_POS = _EnumInfo('joint', 1, 'default_state', component=4, keyword='dof_positions')
    JOINT_MAX_EFFORT = _EnumInfo('joint', 1, 'dof_max_efforts')
    JOINT_MAX_VELOCITY = _EnumInfo('joint', 1, 'dof_max_velocities')
    JOINT_MAX_POS = _EnumInfo('joint', 2, 'dof_limits')  # combination of the lower and the upper limit
    JOINT_ARMATURES = _EnumInfo('joint', 1, 'dof_armatures')
    JOINT_FRICTION = _EnumInfo('joint', 3, 'dof_friction_properties')  # static, dynamic and viscous friction
    JOINT_FRICTION_STATIC = _EnumInfo('joint', 1, 'dof_friction_properties', component=0, keyword='static_frictions')
    JOINT_FRICTION_DYNAMIC = _EnumInfo('joint', 1, 'dof_friction_properties', component=1, keyword='dynamic_frictions')
    JOINT_FRICTION_VISCOUS = _EnumInfo('joint', 1, 'dof_friction_properties', component=2, keyword='viscous_frictions')
    JOINT_MEASURED_EFFORT = _EnumInfo('joint', 1, 'dof_projected_joint_forces')
    SUB_BODY_INERTIA = _EnumInfo('sub_body', 9, 'link_inertias')
    SUB_BODY_MASS = _EnumInfo('sub_body', 1, 'link_masses')
    SUB_BODY_COM = _EnumInfo('sub_body', 7, 'link_coms')  # center of mass
    SUB_BODY_COM_POS = _EnumInfo('sub_body', 3, 'link_coms', component=0, keyword='positions')
    SUB_BODY_COM_ROT = _EnumInfo('sub_body', 4, 'link_coms', component=1, keyword='orientations')

    def __init__(self, category, length, accessor, component=None, keyword=None):
        """
        Constructor.

        Args:
            category (str): The category of the observation type, either 'body', 'joint' or 'sub_body'.
            length (int): The length of the observation, representing the number of values
                associated with this observation type.
            accessor (str): The name of the Isaac Sim call serving the observation, without the ``get_`` or
                ``set_`` prefix.
            component (int, None): Which component of the tuple the getter returns this type is, or None
                when the type takes every component or the getter returns a single array.
            keyword (str, None): The keyword the setter takes that same component under, or None when the
                setter takes the value positionally.

        The keyword selecting the elements to act on within the prim, ``element_kwarg``, follows from the
        category: bodies have no such keyword, joints take ``dof_indices`` and sub bodies ``link_indices``.
        """
        self.category = category
        self.length = length
        self.accessor = accessor
        self.component = component
        self.keyword = keyword
        if self.is_joint():
            self.element_kwarg = 'dof_indices'
        elif self.is_sub_body():
            self.element_kwarg = 'link_indices'
        else:
            self.element_kwarg = None

    def is_body(self):
        """
        Checks whether the observation type belongs to the 'body' category.

        Returns:
            bool: True if the observation type is category 'body', False otherwise.
        """
        return self.category == 'body'

    def is_joint(self):
        """
        Checks whether the observation type belongs to the 'joint' category.

        Returns:
            bool: True if the observation type is category 'joint', False otherwise.
        """
        return self.category == 'joint'

    def is_sub_body(self):
        """
        Checks whether the observation type belongs to the 'sub_body' category, meaning
        it is a property of a rigid body within an articulation.

        Returns:
            bool: True if the observation type is category 'sub_body', False otherwise.
        """
        return self.category == 'sub_body'


class ObservationHelper:
    """
    A helper class reading and writing the properties of the simulated entities.

    It resolves every entry of the observation and the additional data specifications into the Isaac Sim view
    and accessor serving it, assembles the observation of each environment out of those reads, and reports the
    limits of the resulting observation vector.

    """

    def __init__(self, observation_spec, additional_data_spec, num_env, robots, views, env_pos, actuation_helper):
        """
        Constructor.

        Args:
            observation_spec (list): A list containing the names of data that should be made available to the agent as
               an observation and their type (ObservationType). They are combined with a path, which is used
               to access the prim, and a list or a single string with name of the subelements of prim which
               should be accessed. For example a subbody or a joint of an ArticulationView
               An entry in the list is given by: (key, name, type, element). The name can later be used to retrieve
               specific observations.
            additional_data_spec (list): A specification of the same form, for the data read from or written to
                during the simulation without being part of the observation.
            num_env (int): Number of parallel environments.
            robots (Articulation): The articulation covering the robot of every environment.
            views (dict): The view covering every environment, for each path appearing in the specifications.
            env_pos (torch.tensor): The origin of each environment, which world positions are relative to.
            actuation_helper (ActuationHelper): The helper owning the joints the agent controls.
        """
        self._observation_spec = observation_spec
        self._additional_data_spec = additional_data_spec
        self._num_env = num_env

        self._robots = robots
        self._views = views
        self._env_pos = env_pos
        self._actuation_helper = actuation_helper

        self._observers = {}
        self._additionals = {}
        self._consistent_property_storage = {}

        self._obs_low = None
        self._obs_high = None

        self.obs_idx_map = self._compute_obs_idx_map()
        self.obs_types_idx_map = self._compute_type_idx_map()

    def set_up(self):
        """
        Resolves the specifications into the accessors serving them, and reapplies the properties that survive a
        reset. Isaac Sim only names the joints and the bodies of a prim once the simulation is running, so this
        cannot happen at construction.

        """
        self._observers = self._create_observer_tuple(self._observation_spec)
        self._additionals = self._create_observer_tuple(self._additional_data_spec)

        self.reapply_consistent_properties()

    def compute_obs_limits(self):
        """
        Computes and stores the lower and upper limits of the observation. They are the ones the simulator
        declares for the observed joints, and unbounded for every other observation type.

        Returns:
            Two tensors: the first contains the lower limit, and the second contains the upper limit.

        """
        obs_low = []
        obs_high = []
        obs = self._read_observations()

        for name, (_, obs_type, joint_index) in self._observers.items():
            obs_count = obs[name][0, ...].numel()

            if obs_type == ObservationType.JOINT_POS:
                lower, upper = self._robots.get_dof_limits()
                joints = wp.to_torch(joint_index).long()
                obs_low.append(wp.to_torch(lower)[0, joints].to(TorchUtils.get_device()))
                obs_high.append(wp.to_torch(upper)[0, joints].to(TorchUtils.get_device()))
            elif obs_type == ObservationType.JOINT_VEL:
                limit = self._robots.get_dof_max_velocities(indices=[0], dof_indices=joint_index)
                limit = wp.to_torch(limit)[0]
                obs_low.append(-limit)
                obs_high.append(limit)
            else:
                obs_low.append(torch.full((obs_count, ), -torch.inf, device=TorchUtils.get_device()))
                obs_high.append(torch.full((obs_count, ), torch.inf, device=TorchUtils.get_device()))

        self._obs_low = torch.concatenate(obs_low)
        self._obs_high = torch.concatenate(obs_high)

        return self.obs_limits

    def observe(self):
        """
        Reads every observation from the simulation and assembles them into the observation of each environment.

        Returns:
            A tensor of shape (num_env, combined length of all observations).

        """
        return self.build_obs(self._read_observations())

    def build_obs(self, data):
        """
        Constructs a tensor for all observations and fills it with the data of the dictionary.

        Args:
            data (dict): A dictionary where keys are observations names and values are their corresponding
                data tensors. Should contain all data for all observations specified in
                observation_spec

        Returns:
            A tensor of shape (num_env, combined length of all observations), containing all
            data given in the dictionary.
        """
        size = self.obs_length
        obs = torch.zeros(self._num_env, size, device=TorchUtils.get_device())

        for name, indices in self.obs_idx_map.items():
            if name in data:
                obs[:, indices] = data[name]

        return obs

    def get_from_obs(self, obs, name):
        """
        Retrieves the data from obs that corresponds to the observation specified by name.

        Args:
            obs (torch.tensor): The observation tensor.
            name (str): The name specifing the specific observation to extract.

        Returns:
            A tensor containing the specified observation data
        """
        indices = self.obs_idx_map[name]
        return obs[:, indices]

    def get_by_type_from_obs(self, obs, obs_type):
        """
        Retrieves all data from obs of a specific observation type.

        Args:
            obs (torch.tensor): The observation tensor.
            obs_type (ObservationType): The type of observation to retrieve.

        Returns:
            A tensor containing the specified observation data
        """
        indices = self.obs_types_idx_map[obs_type]
        return obs[:, indices]

    def add_obs(self, name, length, min_value, max_value):
        """
        Adds a new observation to the observation space.

        Args:
            name (str): The name of the new observation
            length (int): The length of the observation
            min_value (float, torch.tensor): The lower bound for the observation.
            max_value (float, torch.tensor): The upper bound for the observation
        """
        idx = self.obs_length
        self.obs_idx_map[name] = torch.tensor(list(range(idx, idx + length)), device=TorchUtils.get_device())

        self._obs_low = torch.concatenate([self._obs_low, self._as_bound(min_value, length)])
        self._obs_high = torch.concatenate([self._obs_high, self._as_bound(max_value, length)])

    def read_data(self, name, env_indices=None):
        """
        Read data from isaac sim.

        Args:
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.
            env_indices (torch.tensor, None): The environments to read from, all of them if None.

        Returns:
            The desired data as a tensor.

        """
        view, obs_type, element_idx = self._find_accessor(name)
        return self.read_property(view, obs_type, element_idx=element_idx, env_indices=env_indices)

    def write_data(self, name, value, env_indices=None, reapply_after_reset=False):
        """
        Writes data to isaac sim.

        Args:
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.
            value (torch.tensor): The data that should be written.
            env_indices (torch.tensor, None): The environments to write to, all of them if None.
            reapply_after_reset (bool): Whether the written property should be reapplied after a world reset.
                Defaults to False.

        """
        view, obs_type, element_idx = self._find_accessor(name)
        self.set_property(view, obs_type, value, element_idx=element_idx, env_indices=env_indices)

        if reapply_after_reset:
            self._consistent_property_storage[name] = lambda: self.set_property(
                view, obs_type, value.clone(), element_idx=element_idx, env_indices=env_indices
            )

    def set_joint_data(self, value, type, element_idx=None, env_indices=None):
        """
        Sets the joint properties for the specified joints and environments.

        Args:
            value (torch.tensor): The value to set for the specified joint property.
            type (mushroom_rl.utils.isaac_sim.ObservationType): The type of joint property to be set
                (e.g., "position", "velocity").
            element_idx (torch.tensor, list[int], optional): The indices of the joints to update.
                If None, defaults to the controlled joints.
            env_indices (torch.tensor, list[int], optional): The indices of the environments where
                the joint data should be updated. If None, applies to all environments.

        """
        if element_idx is None and type.is_joint():
            element_idx = self._actuation_helper.controlled_dofs
        self.set_property(self._robots, type, value, element_idx, env_indices)

    def read_property(self, view, obs_type, element_idx=None, env_indices=None):
        """
        Retrieves a specific property from the given view based on the observation type.

        The getter the observation type names is called once, and only the types whose value is not one of
        the arrays it returns are post-processed explicitly.

        Args:
            view: The view object that provides access to simulation data.
            obs_type (ObservationType): The type of observation to retrieve.
            element_idx (warp.array, None): Indices of the joints or bodies for which to retrieve data, in
                the form Isaac Sim returns them from ``get_dof_indices`` and ``get_link_indices``. Only used
                for joint- and sub-body-related observation types.
            env_indices (torch.Tensor, None): Indices of the environments for which to retrieve data.

        Returns:
            The requested property as a torch tensor.

        """
        indices = None if env_indices is None else wp.from_torch(env_indices.to(torch.int32))

        if obs_type == ObservationType.JOINT_MAX_POS:
            # the getter reads every joint of the articulation, the observation keeps the controlled ones
            lower, upper = view.get_dof_limits()
            dof_limits = torch.stack((wp.to_torch(lower), wp.to_torch(upper)), dim=2)
            return dof_limits[:, self._actuation_helper.controlled_joints]

        selection = self._selection(obs_type, indices, element_idx)
        data = getattr(view, f'get_{obs_type.accessor}')(**selection)

        if obs_type == ObservationType.BODY_POS:
            env_pos = self._env_pos if env_indices is None else self._env_pos[env_indices]
            return wp.to_torch(data[obs_type.component]) - env_pos
        elif obs_type in (ObservationType.JOINT_GAIN, ObservationType.JOINT_FRICTION):
            # one scalar per joint each, so the components stack into a new trailing axis
            return torch.stack([wp.to_torch(component) for component in data], dim=-1)
        elif obs_type in (ObservationType.BODY_VEL, ObservationType.SUB_BODY_COM):
            # vectors each, so the components concatenate along the trailing axis
            return torch.cat([wp.to_torch(component) for component in data], dim=-1)

        return wp.to_torch(data if obs_type.component is None else data[obs_type.component])

    def set_property(self, view, obs_type, value, element_idx=None, env_indices=None):
        """
        Sets the specified property values immediately.

        Only the types whose value has to be split over several arguments of the setter the observation type
        names are handled explicitly; the others are passed to it whole.

        Args:
            view: The isaac sim view where the properties should be set.
            obs_type (ObservationType): The type of observation to update.
            value (torch.Tensor): The new values to be assigned.
            element_idx (warp.array, None): The joint or body indices to be updated, in the form Isaac Sim
                returns them from ``get_dof_indices`` and ``get_link_indices``.
            env_indices (torch.Tensor, None): The environment indices to apply the update.

        """
        if obs_type == ObservationType.JOINT_MEASURED_EFFORT:
            raise NotImplementedError("Set function for measured effort doesn't exist in isaacsim.core.")

        indices = None if env_indices is None else wp.from_torch(env_indices.to(torch.int32))
        selection = self._selection(obs_type, indices, element_idx)
        setter = getattr(view, f'set_{obs_type.accessor}')

        if obs_type == ObservationType.BODY_POS:
            setter(positions=wp.from_torch(value + self._env_pos[env_indices]), **selection)
        elif obs_type == ObservationType.BODY_VEL:
            setter(wp.from_torch(value[:, :3]), wp.from_torch(value[:, 3:]), **selection)
        elif obs_type == ObservationType.JOINT_GAIN:
            # stiffness is the proportional gain, damping the derivative one
            setter(stiffnesses=wp.from_torch(value[:, :, 0]), dampings=wp.from_torch(value[:, :, 1]), **selection)
        elif obs_type == ObservationType.JOINT_MAX_POS:
            setter(lower=wp.from_torch(value[..., 0]), upper=wp.from_torch(value[..., 1]), **selection)
        elif obs_type == ObservationType.JOINT_FRICTION:
            setter(wp.from_torch(value[..., 0]), wp.from_torch(value[..., 1]), wp.from_torch(value[..., 2]),
                   **selection)
        elif obs_type == ObservationType.SUB_BODY_COM:
            setter(positions=wp.from_torch(value[..., :3]), orientations=wp.from_torch(value[..., 3:]), **selection)
        elif obs_type.keyword is None:
            setter(wp.from_torch(value), **selection)
        else:
            setter(**{obs_type.keyword: wp.from_torch(value)}, **selection)

    def clear_consistent_properties(self, names=None):
        """
        Removes properties from the consistent property storage.

        Args:
            names (list[str], None): List of property names to remove.
                If None, removes all keys from the consistent property storage.

        """
        if names is None:
            self._consistent_property_storage.clear()
        else:
            for name in names:
                self._consistent_property_storage.pop(name, None)

    def reapply_consistent_properties(self):
        """
        Reapplies the properties that were written with ``reapply_after_reset``.

        """
        for reapply_data in self._consistent_property_storage.values():
            reapply_data()

    @property
    def obs_limits(self):
        return self._obs_low, self._obs_high

    @property
    def obs_length(self):
        return max(map(lambda x: x[-1].item(), self.obs_idx_map.values())) + 1

    def _create_observer_tuple(self, spec):
        """
        Resolves a specification into the view, observation type and element indices used to access it.

        Args:
            spec (list): The observation or additional data specification.

        Returns:
            A dictionary mapping each name to its ``(view, obs_type, element_idx)`` accessor.

        """
        mapping = {}
        for name, path, obs_type, element_names in spec:
            if isinstance(element_names, str):
                element_names = [element_names]

            view = self._views[path]
            element_idx = None

            accessor = f'get_{obs_type.accessor}'
            assert hasattr(view, accessor), f'{type(view).__name__} has no {accessor}, required by {obs_type}.'

            # Find indices of joints or bodies
            if obs_type.is_joint() or obs_type.is_sub_body():
                index_fn = view.get_dof_indices if obs_type.is_joint() else view.get_link_indices
                element_idx = index_fn(element_names)

            mapping[name] = (view, obs_type, element_idx)
        return mapping

    def _find_accessor(self, name):
        """
        Looks up the accessor registered for a name, in the additional data first and the observations after.

        Args:
            name (str): A name referring to an entry contained in additional_data_spec or observation_spec.

        Returns:
            The ``(view, obs_type, element_idx)`` accessor for that name.

        """
        if name in self._additionals:
            return self._additionals[name]

        return self._observers[name]

    def _read_observations(self):
        """
        Retrieves the current observations from the simulation.

        Returns:
            A dictionary mapping observation names to their respective values

        """
        obs = {}
        for name, (view, obs_type, element_idx) in self._observers.items():
            obs[name] = self.read_property(view, obs_type, element_idx=element_idx)
        return obs

    def _as_bound(self, value, length):
        """
        Brings an observation bound into tensor form, broadcasting scalars over the observation length.

        Args:
            value (float, torch.tensor): The bound, either a single value or one value per element.
            length (int): The length of the observation.

        Returns:
            A tensor of shape (length, ) holding the bound.

        """
        if torch.is_tensor(value):
            return value.to(TorchUtils.get_device())
        elif hasattr(value, "__len__"):
            return torch.tensor(value, device=TorchUtils.get_device())

        return torch.full((length, ), value, device=TorchUtils.get_device())

    def _compute_obs_idx_map(self):
        """
        Computes a mapping from observation names to index ranges.

        Returns:
            A dictionary mapping observation names to lists of index values.
        """
        index = 0
        mapping = {}
        for name, _, obs_type, element_names in self._observation_spec:
            n_elements = len(element_names) if isinstance(element_names, list) else 1
            end_index = index + obs_type.length * n_elements
            mapping[name] = torch.tensor(list(range(index, end_index)), device=TorchUtils.get_device())
            index = end_index
        return mapping

    def _compute_type_idx_map(self):
        """
        Computes a mapping from observation types to their corresponding indices.

        Returns:
            A dictionary mapping ObservationType instances to lists of index values.
        """
        mapping = {}
        for obs_type in ObservationType.__members__.values():
            names = [x[0] for x in self._observation_spec if x[2] == obs_type]
            indices = []
            for name in names:
                indices.extend(self.obs_idx_map[name])
            mapping[obs_type] = indices
        mapping = {key: torch.tensor(value, device=TorchUtils.get_device()) for key, value in mapping.items()}
        return mapping

    @staticmethod
    def _selection(obs_type, indices, element_idx):
        """
        Builds the keywords selecting what a getter or a setter acts on.

        Args:
            obs_type (ObservationType): The type of observation being accessed.
            indices (warp.array, None): The environments to act on, None for all of them.
            element_idx (warp.array, None): The joints or bodies to act on, None for all of them.

        Returns:
            A dictionary of keyword arguments.

        """
        if obs_type.element_kwarg is None:
            return dict(indices=indices)

        return {'indices': indices, obs_type.element_kwarg: element_idx}
