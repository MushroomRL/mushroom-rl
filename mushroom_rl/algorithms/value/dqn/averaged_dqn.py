from copy import deepcopy

from mushroom_rl.algorithms.value.dqn import AbstractDQN
from mushroom_rl.approximators.parametric import TorchApproximator


class AveragedDQN(AbstractDQN):
    """
    Averaged-DQN algorithm.
    "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning".
    Anschel O. et al. 2017.

    """
    def __init__(self, mdp_info, policy, approximator=TorchApproximator, n_approximators=5, **params):
        """
        Constructor.

        Args:
            approximator (class, TorchApproximator): the approximator to use to fit the Q-function;
            n_approximators (int, 5): the number of target approximators to store.

        """
        assert n_approximators > 1

        params['approximator_params'] = dict(params['approximator_params'])
        params['approximator_params']['n_models'] = n_approximators

        super().__init__(mdp_info, policy, approximator, **params)

        self._n_fitted_target_models = 1

        self._add_save_attr(_n_fitted_target_models='primitive')

    def _initialize_regressors(self, approximator, approximator_params):
        # the online approximator is a single model, only the target is an ensemble of n_models
        train_params = deepcopy(approximator_params)
        n_approximators = train_params.pop('n_models')

        self.approximator = approximator(**train_params)
        self.target_approximator = approximator(prediction='all', **deepcopy(approximator_params))

        w = self.approximator.get_weights()
        self.target_approximator.set_weights(w.repeat(n_approximators, 1))

    def _update_target(self):
        n_approximators = len(self.target_approximator)

        idx = self._n_updates // self._target_update_frequency % n_approximators
        self.target_approximator[idx].set_weights(self.approximator.get_weights())

        if self._n_fitted_target_models < n_approximators:
            self._n_fitted_target_models += 1

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        q = q[:self._n_fitted_target_models].mean(0)
        if absorbing.any():
            q *= ~absorbing.unsqueeze(1)

        return q.max(1).values
