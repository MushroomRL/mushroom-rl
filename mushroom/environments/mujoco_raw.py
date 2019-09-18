import mujoco_py
from mujoco_py import functions as mj_fun
import numpy as np
from enum import Enum
from mushroom.environments import Environment, MDPInfo
from mushroom.utils.spaces import Box


class ObservationType(Enum):
    """

    An enum indicating the type of data that should be added to the observation of the environment, can be
    Joint-/Body-/Site- positions and velocities

    """

    __order__ = "BODY_POS BODY_VEL JOINT_POS JOINT_VEL SITE_POS SITE_VEL"
    BODY_POS = 0
    BODY_VEL = 1
    JOINT_POS = 2
    JOINT_VEL = 3
    SITE_POS = 4
    SITE_VEL = 5


class DataMap:
    """

    Helper variable to access the correct arrays in the MuJoCo data structure given the type of data that we want
    to access

    """

    def __init__(self, sim):
        self.sim = sim

    def __call__(self, otype):
        if not isinstance(otype, ObservationType):
            otype = ObservationType(otype)

        if otype == ObservationType.BODY_POS:
            return self.sim.data.body_xpos
        elif otype == ObservationType.BODY_VEL:
            return self.sim.data.body_xvelp
        elif otype == ObservationType.JOINT_POS:
            return self.sim.data.qpos
        elif otype == ObservationType.JOINT_VEL:
            return self.sim.data.qvel
        elif otype == ObservationType.SITE_POS:
            return self.sim.data.site_xpos
        else:
            return self.sim.data.site_xvelp


class MojucoRaw(Environment):

    def __init__(self, file_name, actuation_spec, observation_spec, gamma, horizon, nsubsteps=1,
                 additional_data_spec=None, collision_groups=None):
        """

        Create a mushroom environment using the MuJoCo simulator

        Args:
            file_name (string): The path to the XML file with which the environment should be created
            actuation_spec (list): A list specifying of actuator names (strings) that should be controlled by the agent
            observation_spec (list): A list containing the names of data that should be made available to the agent as
                                     an observation and their type (ObservationType). An entry in the list is given by:
                                     (name, type)
            gamma (float): The discounting factor of the environment
            horizon (int): The maximum horizon for the environment
            nsubsteps (int): The number of substeps to use by the MuJoCo simulator. An action given by the agent will be
                             applied for nsubsteps before the agent receives the next observation and can act
                             accordingly
            additional_data_spec (list): A list containing the data fields of interest, which should be read from or
                                         written to during simulation. The entries are given as the following tuples:
                                         (key, name, type)
                                         key is a string for later referencing in the "read_data" and "write_data"
                                         methods. The name is the name of the object in the XML specification and the
                                         type is the ObservationType
            collision_groups (list): A list containing groups of geoms for which collisions should be checked during
                                     simulation via "check_collision". The entries are given as the following tuples:
                                     (key, geom_names)
                                     key is a string for later referencing in the "check_collision" method.
                                     geom_names is a list of geom names in the XML specification.
        """

        # Create the simulation
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(file_name), nsubsteps=nsubsteps)
        self.viewer = None

        # Create a mapping from ObservationTypes to the corresponding index and data arrays
        self.id_maps = [self.sim.model._body_name2id, self.sim.model._body_name2id, self.sim.model._joint_name2id,
                        self.sim.model._joint_name2id, self.sim.model._site_name2id, self.sim.model._site_name2id]
        self.data_map = DataMap(self.sim)

        # Read the actuation spec and build the mapping between actions and ids as well as their limits
        low = []
        high = []
        self.action_indices = []
        for name in actuation_spec:
            index = self.sim.model._actuator_name2id[name]
            self.action_indices.append(index)
            low.append(self.sim.model.actuator_ctrlrange[index][0])
            high.append(self.sim.model.actuator_ctrlrange[index][1])

        action_space = Box(np.array(low), np.array(high))

        # Read the number of kinds of observations
        n_obs = [0] * len(ObservationType)
        for otype in ObservationType:
            n_obs[otype.value] = int(np.sum([1 if ot == otype else 0 for __, ot in observation_spec]))

        # Pre-compute the offsets using this information
        offsets = [0]
        for i in range(1, len(ObservationType)):
            if i - 1 == ObservationType.JOINT_VEL.value or i - 1 == ObservationType.JOINT_POS.value:
                mul = 1
            else:
                mul = 3
            offsets.append(offsets[i - 1] + mul * n_obs[i - 1])
            # n_obs[i] += n_obs[i - 1]

        # Read the observation spec and build the mapping to quickly assemble the observations in every step. It is
        # ensured that the values appear in the order they are specified
        low = []
        high = []
        self.observation_indices = []
        self.observation_sub_indices = {}
        for name, ot in observation_spec:
            if ot.value not in self.observation_sub_indices:
                indices = []
                self.observation_sub_indices[ot.value] = indices
            else:
                indices = self.observation_sub_indices[ot.value]

            # Depending on the type of the observation, we need to add multiple entries in the indices list
            if ot == ObservationType.JOINT_POS or ot == ObservationType.JOINT_VEL:
                self.observation_indices.append(offsets[ot.value] + len(indices))
            else:
                self.observation_indices.extend([offsets[ot.value] + 3 * len(indices) + i for i in range(0, 3)])
            indices.append(self.id_maps[ot.value][name])

            # We can only specify limits for the joint positions, all other information can be potentially unbounded
            if ot == ObservationType.JOINT_POS:
                joint_range = self.sim.model.jnt_range[indices[-1]]
                if joint_range[0] == joint_range[1] == 0.0:
                    low.append(-np.inf)
                    high.append(np.inf)
                else:
                    low.append(joint_range[0])
                    high.append(joint_range[1])
            elif ot == ObservationType.JOINT_VEL:
                low.append(-np.inf)
                high.append(np.inf)
            else:

                low.extend([-np.inf] * 3)
                high.extend([np.inf] * 3)

        observation_space = Box(np.array(low), np.array(high))

        # Pre-process the additional data to allow for fast writing and reading to and from arrays in MuJoCo
        self.additional_data = {}
        for key, name, ot in additional_data_spec:
            self.additional_data[key] = (ot.value, self.id_maps[ot.value][name])

        # Pre-process the collision groups for "fast" detection of contacts
        self.collision_groups = {}
        if self.collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {self.sim.model._geom_name2id[geom_name] for geom_name in geom_names}

        # Finally, we create the MDP information and call the constructor of the parent class
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        mj_fun.mj_resetData(self.sim.model, self.sim.data)
        self.setup()
        return self._create_observation()

    def _create_observation(self):
        """

        Creates the observation of the environment using the information passed in the constructor

        Returns:
            An observation containing the information specified in the constructor as a one-dimensional numpy array

        """

        observation = []
        for i in range(0, len(ObservationType)):
            if i in self.observation_sub_indices:
                observation.append(self.data_map(i)[self.observation_sub_indices[i]].reshape(-1))

        return np.concatenate(observation)[self.observation_indices]

    def step(self, action):
        cur_obs = self._create_observation()

        # The clipping of the outputs is done by MuJoCo for us
        self.sim.data.ctrl[self.action_indices] = action

        self.sim.step()

        next_obs = self._create_observation()

        # Do we pass the clipped or non-clipped action?
        reward = self.reward(cur_obs, action, next_obs)

        return next_obs, reward, self.is_absorbing(next_obs), {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.viewer.render()

    def read_data(self, name):
        """

        Reads data form the MuJoCo data structure

        Args:
            name (string): A name referring to an entry contained the additional_data_spec list handed to the
                           constructor

        Returns:
            The desired data as a one-dimensional numpy array

        """
        data_id, id = self.additional_data[name]
        return self.data_map(data_id)[id]

    def write_data(self, name, value):
        """

        Writes data to the MuJoCo data structure

        Args:
            name (string): A name referring to an entry contained in the additional_data_spec list handed to the
                           constructor
            value (ndarray): The data that should be written

        """
        data_id, id = self.additional_data[name]
        self.data_map(data_id)[id][:] = value

    def check_collision(self, group1, group2):
        """

        Checks for collision between the specified groups

        Args:
            group1 (string): A name referring to an entry contained in the collision_groups list handed to the
                             constructor
            group2 (string): A name referring to an entry contained in the collision_groups list handed to the
                             constructor

        Returns:
            A flag indicating whether a collision occurred between the given groups or not
        """
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        for coni in range(0, self.sim.data.ncon):
            con = self.sim.data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True
        return False

    def stop(self):
        v = self.viewer
        self.viewer = None
        del v

    def reward(self, state, action, next_state):
        """
        Compute the reward based on the given transition

        Args:
            state (np.array): the current state of the system
            action (np.array): the action that is applied in the current state
            next_state (np.array): the state reached after applying the given action

        Returns:
            The reward as a floating point scalar value

        """
        raise NotImplementedError

    def is_absorbing(self, state):
        """
        Check whether the given state is an absorbing state or not

        Args:
            state (np.array): the state of the system

        Returns:
            A boolean flag indicating whether this state is absorbing or not
        """
        raise NotImplementedError

    def setup(self):
        """
        A function that allows to execute setup code after an environment reset
        """
        pass
