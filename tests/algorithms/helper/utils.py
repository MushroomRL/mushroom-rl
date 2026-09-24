import torch
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import itertools

import mushroom_rl
from mushroom_rl.core import MDPInfo, AgentInfo, Dataset
from mushroom_rl.core.spaces import Discrete
from mushroom_rl.policy.td_policy import TDPolicy
from mushroom_rl.policy.torch_policy import TorchPolicy
from mushroom_rl.policy.policy import HasWeights
from mushroom_rl.policy.noise_policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.distributions.gaussian import GaussianDiagonalDistribution
from mushroom_rl.approximators.table import Table
from mushroom_rl.approximators.approximator import Approximator, Ensemble
from mushroom_rl.approximators.q_approximator import QApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory, PrioritizedReplayMemory
from mushroom_rl.rl_utils.parameters import Parameter, VariableParameter, LinearParameter, DecayParameter, \
    VarianceParameter, WindowedVarianceParameter
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer, SGDOptimizer, AdamOptimizer

from mushroom_rl.features._impl import TilesFeatures, FunctionalFeatures, BasisFeatures


class TestUtils:

    @classmethod
    def assert_eq(cls, this, that):
        """
        Check and compare two objects for equality
        """
        for check, kind, compare in cls._eq_rules():
            if check(this, that, kind):
                compare(this, that)
                return
        assert this == that

    @classmethod
    def _eq_rules(cls):
        """
        The ordered ``(check, type, compare)`` rules of :meth:`assert_eq`: the first rule whose check holds compares
        the two objects.
        """
        def asserting(equal):
            def compare(this, that):
                assert equal(this, that)
            return compare

        def pairwise(this, that):
            assert len(this) == len(that)
            for a, b in zip(this, that):
                cls.assert_eq(a, b)

        def mapping(this, that):
            assert this.keys() == that.keys()
            pairwise(list(this.values()), list(that.values()))

        def ensemble(this, that):
            assert len(this) == len(that)
            pairwise(this._models, that._models)

        def ignored(this, that):
            pass

        def both_callable(this, that, kind):
            return callable(this) and callable(that)

        exact, sub = cls._check_type, cls._check_subtype
        return [
            (exact, list, pairwise),
            (exact, dict, mapping),
            (sub, Ensemble, ensemble),
            (sub, QApproximator, lambda a, b: pairwise(a._models, b._models)),
            (exact, Table, asserting(lambda a, b: cls._eq_numpy(a.table, b.table))),
            (sub, Approximator, asserting(cls.eq_weights)),
            (sub, TorchPolicy, asserting(cls.eq_weights)),
            (sub, HasWeights, asserting(cls.eq_weights)),
            (sub, TDPolicy, lambda a, b: cls.assert_eq(a.get_q(), b.get_q())),
            (exact, torch.optim.Optimizer, asserting(lambda a, b: cls.eq_save_dict(a.state_dict(), b.state_dict()))),
            (exact, itertools.chain, asserting(cls.eq_chain)),
            (exact, MDPInfo, asserting(cls.eq_mdp_info)),
            (exact, AgentInfo, asserting(cls.eq_agent_info)),
            (exact, Dataset, asserting(cls.eq_dataset)),
            (exact, PrioritizedReplayMemory, asserting(cls.eq_prioritized_replay_memory)),
            (exact, ReplayMemory, asserting(cls.eq_replay_memory)),
            (exact, OrnsteinUhlenbeckPolicy, asserting(cls.eq_ornstein_uhlenbeck_policy)),
            (exact, TilesFeatures, asserting(cls.eq_tiles_features)),
            (exact, LinearParameter, asserting(cls.eq_linear_parameter)),
            (exact, DecayParameter, asserting(cls.eq_decay_parameter)),
            (exact, WindowedVarianceParameter, asserting(cls.eq_windowed_variance_parameter)),
            (exact, VarianceParameter, asserting(cls.eq_variance_parameter)),
            (exact, VariableParameter, asserting(cls.eq_variable_parameter)),
            (exact, Parameter, asserting(cls.eq_parameter)),
            (exact, AdaptiveOptimizer, asserting(cls.eq_adaptive_optimizer)),
            (exact, SGDOptimizer, asserting(cls.eq_sgd_optimizer)),
            (exact, AdamOptimizer, asserting(cls.eq_adam_optimizer)),
            (exact, GaussianDiagonalDistribution, asserting(cls.eq_gaussian_diagonal_dist)),
            (exact, Discrete, asserting(cls.eq_discrete)),
            (exact, FunctionalFeatures, asserting(cls._eq_functional_features)),
            (exact, BasisFeatures, asserting(cls._eq_basis_features)),
            (exact, ExtraTreesRegressor, ignored),
            (both_callable, None, ignored),
            (exact, torch.nn.parameter.Parameter, asserting(cls._eq_torch)),
            (exact, torch.Tensor, asserting(cls._eq_torch)),
            (exact, np.ndarray, asserting(cls._eq_numpy)),
        ]

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
        # params contains Tensor Ids which change after loading into a new optimizer instance
        # ref: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html
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
        if isinstance(this.observation_space, mushroom_rl.core.spaces.Box):
            res &= cls.eq_box(this.observation_space, that.observation_space)
        elif isinstance(this.observation_space, mushroom_rl.core.spaces.Discrete):
            res = cls.eq_discrete(this.observation_space, that.observation_space)
        else:
            raise TypeError('Type not supported')

        if isinstance(this.action_space, mushroom_rl.core.spaces.Box):
            res &= cls.eq_box(this.action_space, that.action_space)
        elif isinstance(this.action_space, mushroom_rl.core.spaces.Discrete):
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

        res = this.env_backend == that.env_backend
        res &= this.agent_backend == that.agent_backend
        res &= this.env_device == that.env_device
        res &= this.agent_device == that.agent_device
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

        res = type(this) is type(that)
        res &= this._dataset_info.env_array_backend == that._dataset_info.env_array_backend
        res &= cls.eq_dataset_info(this._dataset_info, that._dataset_info)
        res &= len(this) == len(that) and this.capacity == that.capacity

        columns = ['state', 'action', 'reward', 'next_state', 'absorbing', 'last']
        if this.is_stateful or that.is_stateful:
            columns += ['policy_state', 'policy_next_state']
        for column in columns:
            res &= cls._eq_value(getattr(this, column), getattr(that, column))
        if hasattr(this, 'mask'):
            res &= cls._eq_value(this.mask, that.mask)
            res &= cls._eq_value(this._tail_open, that._tail_open) and this._consumed == that._consumed

        res &= cls._eq_value(this.info, that.info)
        res &= cls._eq_value(this.episode_info, that.episode_info)
        res &= cls._eq_value(this.theta_list, that.theta_list)

        res &= cls._eq_layout(this._layout, that._layout)
        if hasattr(this.history_state, '_slots'):
            res &= cls._eq_value(this.history_state._slots, that.history_state._slots)
        else:
            res &= cls._eq_value(this.history_state.positions, that.history_state.positions)
            res &= cls._eq_value(this.history_state._windows, that.history_state._windows)

        return res

    @classmethod
    def _eq_layout(cls, this, that):
        """
        Compare the row structure of two datasets
        """
        res = type(this) is type(that) and len(this) == len(that)
        res &= this.first == that.first and this.n_joins == that.n_joins
        res &= this.open_heads == that.open_heads and this.open_tails == that.open_tails
        res &= this.pending_heads() == that.pending_heads()
        if hasattr(this, 'array'):
            res &= cls._eq_value(this.array(), that.array())
        if hasattr(this, 'write_head'):
            res &= this.max_size == that.max_size and this.write_head == that.write_head and this.full == that.full
            res &= this._ring_tails == that._ring_tails
            res &= cls._eq_value(this.links, that.links)
        return res

    @classmethod
    def _eq_value(cls, this, that):
        """
        Compare two values that may be nested dictionaries, lists or tuples of arrays
        """
        if this is None or that is None:
            return this is None and that is None
        if isinstance(this, dict):
            return isinstance(that, dict) and this.keys() == that.keys() and \
                all(cls._eq_value(this[key], that[key]) for key in this if not key.startswith('_add'))
        if isinstance(this, (list, tuple)) and isinstance(that, (list, tuple)) and \
                not (len(this) > 0 and np.isscalar(this[0])):
            return len(this) == len(that) and all(cls._eq_value(a, b) for a, b in zip(this, that))
        if isinstance(this, torch.Tensor) or isinstance(that, torch.Tensor):
            return isinstance(this, torch.Tensor) and isinstance(that, torch.Tensor) and \
                this.device == that.device and torch.equal(this, that)
        if isinstance(this, (np.ndarray, list, tuple)) or isinstance(that, (np.ndarray, list, tuple)):
            return np.array_equal(np.asarray(this), np.asarray(that), equal_nan=True)
        return this == that

    @classmethod
    def eq_replay_memory(cls, this, that):
        """
        Compare two ReplayMemory objects for equality
        """
        res = this._initial_size == that._initial_size
        res &= this._max_size == that._max_size
        res &= cls.eq_mdp_info(this._mdp_info, that._mdp_info)
        res &= cls.eq_agent_info(this._agent_info, that._agent_info)
        res &= this._dataset.write_head == that._dataset.write_head
        res &= this._dataset.full == that._dataset.full

        if this._dataset is not None and that._dataset is not None:
            res &= cls.eq_dataset(this._dataset, that._dataset)

        return res

    @classmethod
    def eq_prioritized_replay_memory(cls, this, that):
        """
        Compare two PrioritizedReplayMemory objects for equality
        """

        res = cls.eq_replay_memory(this, that)
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
        res &= cls._eq_numpy(this._masked, that._masked)
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
        if this._dim is not None and that._dim is not None:
            for a, b in zip(this._dim, that._dim):
                res &= a == b
        return res

    @classmethod
    def eq_parameter(cls, this, that):
        """
        Compare two Parameter objects for equality
        """

        res = this._initial_value == that._initial_value
        res &= this._shape == that._shape
        res &= this._log_full == that._log_full
        res &= this._backend == that._backend
        return res

    @classmethod
    def eq_variable_parameter(cls, this, that):
        """
        Compare two VariableParameter objects for equality
        """

        res = cls.eq_parameter(this, that)
        res &= this._min_value == that._min_value
        res &= this._max_value == that._max_value
        res &= cls._eq_numpy(this._n_updates.table, that._n_updates.table)
        return res

    @classmethod
    def eq_linear_parameter(cls, this, that):
        """
        Compare two LinearParameter objects for equality
        """

        res = cls.eq_variable_parameter(this, that)
        res &= this._coeff == that._coeff
        return res

    @classmethod
    def eq_decay_parameter(cls, this, that):
        """
        Compare two DecayParameter objects for equality
        """

        res = cls.eq_variable_parameter(this, that)
        res &= this._exp == that._exp
        return res

    @classmethod
    def eq_variance_parameter(cls, this, that):
        """
        Compare two VarianceParameter objects for equality
        """

        res = cls.eq_variable_parameter(this, that)
        res &= this._exponential == that._exponential
        res &= this._tol == that._tol
        res &= cls._eq_numpy(this._weights_var.table, that._weights_var.table)
        res &= cls._eq_numpy(this._x.table, that._x.table)
        res &= cls._eq_numpy(this._x2.table, that._x2.table)
        res &= cls._eq_numpy(this._parameter_value.table, that._parameter_value.table)
        return res

    @classmethod
    def eq_windowed_variance_parameter(cls, this, that):
        """
        Compare two WindowedVarianceParameter objects for equality
        """

        res = cls.eq_variable_parameter(this, that)
        res &= this._exponential == that._exponential
        res &= this._tol == that._tol
        res &= this._window == that._window
        res &= cls._eq_numpy(this._weights_var.table, that._weights_var.table)
        res &= cls._eq_numpy(this._samples.table, that._samples.table)
        res &= cls._eq_numpy(this._index.table, that._index.table)
        res &= cls._eq_numpy(this._parameter_value.table, that._parameter_value.table)
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
        return issubclass(type(this), check_type) and issubclass(type(that), check_type) and type(this) is type(that)

    @staticmethod
    def _eq_numpy(this, that):
        return np.array_equal(this, that)

    @staticmethod
    def _eq_torch(this, that):
        return torch.equal(this, that)
