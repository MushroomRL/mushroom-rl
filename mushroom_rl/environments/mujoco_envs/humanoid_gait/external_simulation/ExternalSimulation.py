import numpy as np


class ExternalSimulationInterface:
    """
    Allows controling the simulation through external stimulus
    - with default values works as if there was no external stimulus
    """
    def __init__(self, sim):
        self.sim = sim

    def get_action_space(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def reward(self, state, action, next_state):
        raise NotImplementedError

    def is_absorving(self, state):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def update_state(self):
        pass

    def reset(self):
        pass

    def preprocess_action(self, action):
        return action

    def initialize_internal_states(self, state, action):
        pass

    def external_stimulus_to_joint_torques(self, stimu):
        return stimu


class NoExternalSimulation(ExternalSimulationInterface):
    """
    Allows controling the simulation through external stimulus
    - with default values works as if there was no external stimulus
    """

    def __init__(self, sim):
        super().__init__(sim)

    def get_action_space(self):
        return np.array([]), np.array([])

    def get_observation_space(self):
        return np.array([]), np.array([])

    def reward(self, state, action, next_state):
        return 0

    def is_absorving(self, state):
        return False

    def get_observation(self):
        return np.array([])