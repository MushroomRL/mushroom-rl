import torch
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import itertools

import mushroom_rl
from mushroom_rl.core import MDPInfo, AgentInfo, DatasetInfo, Dataset
from mushroom_rl.policy.td_policy import TDPolicy
from mushroom_rl.policy.torch_policy import TorchPolicy
from mushroom_rl.policy.policy import ParametricPolicy
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.sac import SACPolicy
from mushroom_rl.rl_utils.replay_memory import ReplayMemory, PrioritizedReplayMemory
from mushroom_rl.approximators.ensemble import Ensemble
from mushroom_rl.approximators._implementations.action_regressor import ActionRegressor
from mushroom_rl.approximators import Regressor
from mushroom_rl.policy.noise_policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.features._implementations.tiles_features import TilesFeatures
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer, SGDOptimizer, AdamOptimizer
from mushroom_rl.distributions.gaussian import GaussianDiagonalDistribution
from mushroom_rl.approximators.table import Table
from mushroom_rl.rl_utils.spaces import Discrete
from mushroom_rl.features._implementations.functional_features import FunctionalFeatures
from mushroom_rl.features._implementations.basis_features import BasisFeatures


class TestUtils:
    
    @classmethod
    def assert_eq(cls, this, that):
        """
        Check and compare two objects for equality
        """
        if cls._check_type(this, that, list):
            for a, b in zip(this, that): cls.assert_eq(a, b)
        elif cls._check_type(this, that, dict):
            for a, b in zip(this.values(), that.values()): cls.assert_eq(a, b)
        elif cls._check_subtype(this, that, Ensemble):
            assert len(this) == len(that)
            for i in range(0, len(this)):
                cls.assert_eq(this[i], that[i])
        elif cls._check_type(this, that, Regressor):
            if cls._check_type(this._impl, that._impl, ActionRegressor):
                this = this.model
                that = that.model
            for i in range(0, len(this)):
                if cls._check_type(this[i], that[i], list) or cls._check_type(this[i], that[i], Ensemble) \
                        or cls._check_type(this[i], that[i], ExtraTreesRegressor):
                    cls.assert_eq(this[i], that[i])
                else:
                    assert cls.eq_weights(this[i], that[i])
        elif cls._check_subtype(this, that, TorchPolicy) or cls._check_type(this, that, SACPolicy) \
                or cls._check_subtype(this, that, ParametricPolicy):
            assert cls.eq_weights(this, that)
        elif cls._check_subtype(this, that, TDPolicy):
            cls.assert_eq(this.get_q(), that.get_q())
        elif cls._check_type(this, that, torch.optim.Optimizer):
            assert cls.eq_save_dict(this.state_dict(), that.state_dict())
        elif cls._check_type(this, that, itertools.chain):
            assert cls.eq_chain(this, that)
        elif cls._check_type(this, that, MDPInfo):
            assert cls.eq_mdp_info(this, that)
        elif cls._check_type(this, that, AgentInfo):
            assert cls.eq_agent_info(this, that)
        elif cls._check_type(this, that, Dataset):
            assert cls.eq_dataset(this, that)
        elif cls._check_type(this, that, ReplayMemory):
            assert cls.eq_replay_memory(this, that)
        elif cls._check_type(this, that, PrioritizedReplayMemory):
            assert cls.eq_prioritized_replay_memory(this, that)
        elif cls._check_type(this, that, OrnsteinUhlenbeckPolicy):
            assert cls.eq_ornstein_uhlenbeck_policy(this, that)
        elif cls._check_type(this, that, TilesFeatures):
            assert cls.eq_tiles_features(this, that)
        elif cls._check_type(this, that, Parameter):
            assert cls.eq_parameter(this, that)
        elif cls._check_type(this, that, LinearParameter):
            assert cls.eq_linear_parameter(this, that)
        elif cls._check_type(this, that, AdaptiveOptimizer):
            assert cls.eq_adaptive_optimizer(this, that)
        elif cls._check_type(this, that, SGDOptimizer):
            assert cls.eq_sgd_optimizer(this, that)
        elif cls._check_type(this, that, AdamOptimizer):
            assert cls.eq_adam_optimizer(this, that)
        elif cls._check_type(this, that, GaussianDiagonalDistribution):
            assert cls.eq_gaussian_diagonal_dist(this, that)
        elif cls._check_type(this, that, Table):
            assert cls._eq_numpy(this.table, that.table)
        elif cls._check_type(this, that, Discrete):
            assert cls.eq_discrete(this, that)
        elif cls._check_type(this, that, FunctionalFeatures):
            assert cls._eq_functional_features(this, that)
        elif cls._check_type(this, that, BasisFeatures):
            assert cls._eq_basis_features(this, that)
        elif cls._check_type(this, that, ExtraTreesRegressor):
            pass # Equality of ExtraTreeRegressor is not tested
        elif callable(this) and callable(that):
            pass # Equality of Functions is not tested
        elif cls._check_type(this, that, torch.nn.parameter.Parameter):
            assert cls._eq_torch(this, that)
        elif cls._check_type(this, that, np.ndarray):
            assert cls._eq_numpy(this, that)
        else:
            assert this == that
    
    @classmethod
    def eq_weights(cls, this, that):
        """
        Compare the weights of two objects for equality
        """
        return cls._eq_numpy(this.get_weights(), that.get_weights())
    
    @classmethod
    def eq_box(cls, this, that):
        """
        Compare two Box objects for equality
        """
        return cls._eq_numpy(this.low, that.low) and cls._eq_numpy(this.high, that.high) and this.shape == that.shape
    
    @classmethod
    def eq_discrete(cls, this, that):
        """
        Compare two Discrete objects for equality
        """
        return cls._eq_numpy(this.values, that.values) and this.n == that.n

    @classmethod
    def eq_chain(cls, this, that):
        """
        Compare two chain objects for equality
        """
        return list(this) == list(that)

    @classmethod
    def eq_save_dict(cls, this, that):
        """
        Compare two save_dict objects for equality
        """
        this_state, this_param_groups = this.values()
        that_state, that_param_groups = that.values()
        # params contains Tensor Ids which change after loading into a new optimizer instance ref: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html
        del this_param_groups[0]['params']
        del that_param_groups[0]['params']
        res = this_param_groups == that_param_groups
        for t1, t2 in zip(this_state.values(), that_state.values()):
            for v1, v2 in zip(t1.values(), t2.values()):
                if isinstance(v1, torch.Tensor):
                    res &= cls._eq_torch(v1, v2)
                else:
                    res &= v1 == v2
        return res

    @classmethod
    def eq_mdp_info(cls, this, that):
        """
        Compare two mdp_info objects for equality
        """
        res = True
        if isinstance(this.observation_space, mushroom_rl.rl_utils.spaces.Box):
            res &= cls.eq_box(this.observation_space, that.observation_space)
        elif isinstance(this.observation_space, mushroom_rl.rl_utils.spaces.Discrete):
            res = cls.eq_discrete(this.observation_space, that.observation_space)
        else:
            raise TypeError('Type not supported')

        if isinstance(this.action_space, mushroom_rl.rl_utils.spaces.Box):
            res &= cls.eq_box(this.action_space, that.action_space)
        elif isinstance(this.action_space, mushroom_rl.rl_utils.spaces.Discrete):
            res &= cls.eq_discrete(this.action_space, that.action_space)
        else:
            raise TypeError('Type not supported')
            
        res &= this.gamma == that.gamma
        res &= this.horizon == that.horizon
        return res

    @classmethod
    def eq_agent_info(cls, this, that):
        """
                Compare two mdp_info objects for equality
                """
        res = this.is_episodic == that.is_episodic
        res &= this.is_stateful == that.is_stateful
        res &= this.policy_state_shape == that.policy_state_shape
        res &= this.backend == that.backend

        return res

    @classmethod
    def eq_ornstein_uhlenbeck_policy(cls, this, that):
        """
        Compare two OrnsteinUhlenbeckPolicy objects for equality
        """
        
        res = cls.eq_weights(this, that)
        res &= cls._eq_numpy(this._chol_sigma, that._chol_sigma)
        res &= this._theta == that._theta
        res &= this._dt == that._dt
        res &= cls._eq_numpy(this._x0, that._x0)
        res &= cls._eq_numpy(this._x_prev, that._x_prev)
        return res

    @classmethod
    def eq_dataset_info(cls, this, that):
        """
        Compare two dataset classes
        """

        res = this.backend == that.backend
        res &= this.device == that.device
        res &= this.horizon == that.horizon
        res &= this.gamma == that.gamma
        res &= this.state_shape == that.state_shape
        res &= this.state_dtype == that.state_dtype
        res &= this.action_shape == that.action_shape
        res &= this.action_dtype == that.action_dtype
        res &= this.policy_state_shape == that.policy_state_shape
        res &= this.n_envs == that.n_envs

        return res

    @classmethod
    def eq_dataset(cls, this, that):
        """
        Compare two dataset classes
        """

        res = this._array_backend == that._array_backend
        res &= cls.eq_dataset_info(this._dataset_info, that._dataset_info)

        # res &= this._info == that._info TODO fix this equality check
        # res &= this._episode_info == that._episode_info
        # res &= this._theta_list == that._theta_list
        # res &= this._data == that._data

        return res

    @classmethod
    def eq_replay_memory(cls, this, that):
        """
        Compare two ReplayMemory objects for equality
        """
        res = this._initial_size == that._initial_size
        res &= this._max_size == that._max_size
        res &= cls.eq_mdp_info(this._mdp_info, that._mdp_info)
        res &= cls.eq_agent_info(this._agent_info, that._agent_info)
        res &= this._idx == that._idx
        res &= this._full == that._full

        if this._dataset is not None and that._dataset is not None:
            res &= cls.eq_dataset(this._dataset, that._dataset)

        return res

    @classmethod
    def eq_prioritized_replay_memory(cls, this, that):
        """
        Compare two PrioritizedReplayMemory objects for equality
        """
        
        res = this._initial_size == that._initial_size
        res &= this._max_size == that._max_size
        res &= this._alpha == that._alpha
        res &= cls.eq_linear_parameter(this._beta, that._beta)
        res &= this._epsilon == that._epsilon
        res &= cls.eq_sum_tree(this._tree, that._tree)
        return res

    @classmethod
    def eq_sum_tree(cls, this, that):
        """
        Compare two SumTree objects for equality
        """
        
        res = this._max_size == that._max_size
        res &= cls._eq_numpy(this._tree, that._tree)
        res &= cls.eq_dataset(this.dataset, that.dataset)
        res &= this._idx == that._idx
        res &= this._full == that._full
        return res

    @classmethod
    def eq_tiles_features(cls, this, that):
        """
        Compare two TilesFeatures objects for equality
        """
        
        res = this.size == that.size
        for a, b in zip(this._tiles, that._tiles):
            res &= cls.eq_tiles(a, b)
        return res

    @classmethod
    def eq_tiles(cls, this, that):
        """
        Compare two Tiles objects for equality
        """
        
        res = this.size == that.size
        for a, b in zip(this._range, that._range):
            res &= a == b
        for a, b in zip(this._n_tiles, that._n_tiles):
            res &= a == b
        if this._state_components is not None and that._state_components is not None:
            for a, b in zip(this._state_components, that._state_components):
                res &= a == b
        return res

    @classmethod
    def eq_parameter(cls, this, that):
        """
        Compare two Parameter objects for equality
        """
        
        res = this._initial_value == that._initial_value
        res &= this._min_value == that._min_value
        res &= this._max_value == that._max_value
        res &= cls._eq_numpy(this._n_updates.table, that._n_updates.table)
        return res

    @classmethod
    def eq_linear_parameter(cls, this, that):
        """
        Compare two LinearParameter objects for equality
        """
        
        res = cls.eq_parameter(this, that)
        res &= this._coeff == that._coeff
        return res

    @classmethod
    def eq_adaptive_optimizer(cls, this, that):
        """
        Compare two AdaptiveOptimizer objects for equality
        """

        res = cls._eq_numpy(this._eps, that._eps)
        return res

    @classmethod
    def eq_sgd_optimizer(cls, this, that):
        """
        Compare two SGDOptimizer objects for equality
        """

        res = cls._eq_numpy(this._eps, that._eps)
        return res

    @classmethod
    def eq_adam_optimizer(cls, this, that):
        """
        Compare two AdamOptimizer objects for equality
        """

        res = cls._eq_numpy(this._eps, that._eps)
        return res

    @classmethod
    def eq_gaussian_diagonal_dist(cls, this, that):
        """
        Compare two GaussianDiagonalDistribution objects for equality
        """
        
        res = cls._eq_numpy(this.get_parameters(), that.get_parameters())
        return res

    @classmethod
    def _eq_functional_features(cls, this, that):
        """
        Compare two FunctionalFeatures objects for equality
        """
        
        res = this.size == that.size
        return res

    @classmethod
    def _eq_basis_features(cls, this, that):
        """
        Compare two BasisFeatures objects for equality
        """
        
        res = this.size == that.size
        for a, b in zip(this._basis, that._basis):
            res &= str(a) == str(b)
        return res

    @classmethod
    def _eq_listlike(cls, this, that):
        """
        Compare the elements of two listlike objects for equality
        """
        
        res = len(this) == len(that)
        for a, b in zip(this, that):
            if cls._check_type(a, b, np.ndarray):
                res &= cls._eq_numpy(a, b)
            elif cls._check_type(a, b, torch.nn.parameter.Parameter):
                res &= cls._eq_torch(a, b)
            else:
                res &= a == b
        return res

    @staticmethod
    def _check_type(this, that, check_type):
        """
        Check if two object have a specific type
        """
        return isinstance(this, check_type) and isinstance(that, check_type)

    @staticmethod
    def _check_subtype(this, that, check_type):
        """
        Check if two objects have the type of a subclass of a specific type
        """
        return issubclass(type(this), check_type) and issubclass(type(that), check_type) and type(this) == type(that)

    @staticmethod
    def _eq_numpy(this, that):
        return np.array_equal(this, that)

    @staticmethod
    def _eq_torch(this, that):
        return torch.equal(this, that)

