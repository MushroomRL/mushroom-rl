import json
import torch
import pickle
import numpy as np
from copy import deepcopy
from pathlib import Path, PurePath

class Agent(object):
    """
    This class implements the functions to manage the agent (e.g. move the agent
    following its policy).

    """

    def __init__(self, mdp_info, policy, features=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            policy (Policy): the policy followed by the agent;
            features (object, None): features to extract from the state.

        """
        self.mdp_info = mdp_info
        self.policy = policy

        self.phi = features

        self.next_action = None

        self._add_save_attr(
            mdp_info='pickle',
            policy='pickle',
            phi='pickle',
            next_action='numpy')

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError('Agent is an abstract class')

    def draw_action(self, state):
        """
        Return the action to execute in the given state. It is the action
        returned by the policy or the action set by the algorithm (e.g. in the
        case of SARSA).

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """
        if self.phi is not None:
            state = self.phi(state)

        if self.next_action is None:
            return self.policy.draw_action(state)
        else:
            action = self.next_action
            self.next_action = None

            return action

    def episode_start(self):
        """
        Called by the agent when a new episode starts.

        """
        self.policy.reset()

    def stop(self):
        """
        Method used to stop an agent. Useful when dealing with real world
        environments, simulators, or to cleanup environments internals after
        a core learn/evaluate to enforce consistency.

        """
        pass

    def _add_save_attr(self, **attr_dict): # private, put documentation, check if same keys get overwritten
        """
        Adds attributes that should be saved for an agent.

        Args:
            attr_dict (dict): dictionary of attributes mapped to the method that should be used to save and load them

        """
        if not hasattr(self, '_save_attributes'):
            self._save_attributes = dict(_save_attributes='json')
        self._save_attributes.update(attr_dict)

    @classmethod
    def load(cls, path):
        """
        Loads and deserializes the agent from the given location on disk.

        Args:
            path (string): Relative or absolute path to the agents save location.

        """
        if not isinstance(path, str): 
            raise ValueError('path has to be of type string')
        if not Path(path).is_dir():
            raise NotADirectoryError("Path to load agent is not valid")

        # Get algorithm type and save_attributes
        agent_type, save_attributes = cls._load_pickle(PurePath(path, 'agent.config')).values()

        agent = agent_type.__new__(agent_type)

        for att, method in save_attributes.items():
            load_path = Path(path, '{}.{}'.format(att, method))
            
            if load_path.is_file():
                load_method = getattr(cls, '_load_{}'.format(method))
                if load_method is None: raise NotImplementedError('Method _load_{} is not implemented'.format(method))
                att_val = load_method(load_path.resolve())
                setattr(agent, att, att_val)
            else:
                setattr(agent, att, None)
        agent._post_load()
        return agent

    def _post_load(self):
        """
        This method can be overwritten to implement logic that is executed after the loading of the agent.

        """
        pass

    @staticmethod
    def _load_pickle(path):
        with Path(path).open('rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _load_numpy(path):
        with Path(path).open('rb') as f:
            return np.load(f)
    
    @staticmethod
    def _load_torch(path):
        return torch.load(path)
    
    @staticmethod
    def _load_json(path):
        with Path(path).open('r') as f:
            return json.load(f)

    def save(self, path):
        """
        Serialize and save the agent to the given path on disk.

        Args:
            path (string): Relative or absolute path to the agents save location.

        """
        if not isinstance(path, str): raise ValueError('path has to be of type string')

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save algorithm type and save_attributes
        agent_config = dict(
            type=type(self),
            save_attributes=self._save_attributes
        )
        self._save_pickle(PurePath(path, 'agent.config'), agent_config)

        for att, method in self._save_attributes.items():
            attribute = getattr(self, att) if hasattr(self, att) else None
            save_method = getattr(self, '_save_{}'.format(method)) if hasattr(self, '_save_{}'.format(method)) else None
            if attribute is None:
                continue
            elif save_method is None:
                raise NotImplementedError("Method _save_{} is not implemented for class '{}'".format(method, self.__class__.__name__))
            else:
                save_method(PurePath(path, "{}.{}".format(att, method)), attribute)

    @staticmethod
    def _save_pickle(path, obj):
        with Path(path).open('wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def _save_numpy(path, obj):
        with Path(path).open('wb') as f:
            np.save(f, obj)
    
    @staticmethod
    def _save_torch(path, obj):
        torch.save(obj, path)
    
    @staticmethod
    def _save_json(path, obj):
        with Path(path).open('w') as f:
            json.dump(obj, f)

    def copy(self):
        """
        Creates and returns a deepcopy of the agent.
        """
        return deepcopy(self)
