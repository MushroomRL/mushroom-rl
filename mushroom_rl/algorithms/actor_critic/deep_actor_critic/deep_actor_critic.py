from mushroom_rl.core import Agent
from mushroom_rl.utils.torch_utils import TorchUtils


class DeepAC(Agent):
    """
    Base class for deep actor-critic algorithms.

    """
    def __init__(self, mdp_info, policy, actor_optimizer=None, parameters=None, backend='torch', history_length=1,
                 action_history_length=0, device=None):
        """
        Constructor.

        Args:
            actor_optimizer (dict, None): parameters to specify the actor optimizer algorithm; when ``None``, no actor
                optimizer is built;
            parameters (list, None): policy parameters to be optimized by the actor optimizer;
            backend (str, 'torch'): the backend used by the agent;
            history_length (int, 1): number of consecutive observations stacked as policy input;
            action_history_length (int, 0): number of previous actions stacked as policy input;
            device (str, None): device the agent data is stored on, or ``None`` for the default one.

        """
        if actor_optimizer is not None:
            if parameters is not None and not isinstance(parameters, list):
                parameters = list(parameters)
            self._parameters = parameters

            self._optimizer = actor_optimizer['class'](parameters, **actor_optimizer['params'])

            self._clipping = None

            if 'clipping' in actor_optimizer:
                self._clipping = actor_optimizer['clipping']['method']
                self._clipping_params = actor_optimizer['clipping']['params']

        super().__init__(mdp_info, policy, backend=backend, device=device, history_length=history_length,
                         action_history_length=action_history_length)

        self._check_device(policy, 'policy')

        if actor_optimizer is not None:
            self._add_save_attr(
                _optimizer='torch',
                _clipping='torch',
                _clipping_params='pickle'
            )

    def overwrite_actor_parameters(self, parameters):
        """
        Overwrite the parameters updated by the actor optimizer, which default to all the parameters of the
        actor approximator.

        Args:
            parameters (list): the parameters to be updated by the actor optimizer.

        """
        self._parameters = list(parameters)
        if self._optimizer is not None:
            TorchUtils.replace_optimizer_parameters(self._optimizer, self._parameters)

    @property
    def actor_parameters(self):
        """
        Returns:
            The parameters updated by the actor optimizer.

        """
        return self._parameters

    def _optimize_actor_parameters(self, loss):
        """
        Method used to update actor parameters to maximize a given loss.

        Args:
            loss (torch.tensor): the loss computed by the algorithm.

        """
        self._optimizer.zero_grad()
        loss.backward()
        self._clip_gradient()
        self._optimizer.step()

    def _clip_gradient(self):
        if self._clipping:
            self._clipping(self._parameters, **self._clipping_params)


class OnPolicyDeepAC(DeepAC):
    """
    Base class for on policy deep actor-critic algorithms, such as PPO and TRPO.

    """
    def __init__(self, mdp_info, policy, **kwargs):
        """
        Constructor.

        Args:
            **kwargs: parameters forwarded to the ``DeepAC`` constructor.

        """
        self._iter = 1

        super().__init__(mdp_info, policy, **kwargs)

        self._add_save_attr(_iter='primitive')

    def _parse_dataset(self, dataset, get_policy_state=False):
        """
        Update the statistics of the agent preprocessors with the dataset and parse it.

        Args:
            dataset (Dataset): the dataset of the current fit;
            get_policy_state (bool, False): whether the batch also holds the policy state and policy next state of
                each row.

        Returns:
            A tuple ``(batch, state_old)``: the :class:`~mushroom_rl.core.batch.Batch` of the dataset, parsed with the
            updated statistics, and its ``state`` parsed with the statistics before the update.

        """
        state_old = self._history_manager.parse_state(dataset)
        self._history_manager.update_preprocessors(dataset)

        batch = self._history_manager.parse_history(dataset, get_policy_state=get_policy_state)

        return batch, state_old

    def _log_iteration_start(self):
        """
        Open the log block of the current iteration. To be called at the beginning of the fit, so that
        the quantities logged during the fit, e.g. the critic loss, appear under the iteration number.

        """
        if self._logger:
            self._logger.weak_line(debug=True)
            self._logger.debug('Iteration ' + str(self._iter))

    def _log_info(self, dataset, kl, entropy):
        """
        Log the fit information of the current iteration and advance the iteration counter.

        Args:
            dataset (Dataset): the dataset used by the current fit;
            kl (torch.Tensor): the KL divergence between the policy before the fit and the updated policy, or an
                estimate of it computed during the fit;
            entropy (torch.Tensor): the entropy of the policy, or an estimate of it computed during the fit.

        """
        if self._logger:
            self._logger.log_training(R=dataset.compute_J(skip_incomplete=False).mean().item())
            self._logger.log_training('actor', entropy=entropy.item(), kl=kl.item())
            self._logger.advance_step()

        self._iter += 1


class OffPolicyDeepAC(DeepAC):
    """
    Base class for off policy deep actor-critic algorithms, such as DDPG, TD3 and SAC.

    """
    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError('OffPolicyDeepAC is an abstract class')

    def _update_target(self, online, target):
        weights = self._tau() * online.get_weights()
        weights += (1 - self._tau.get_value()) * target.get_weights()
        target.set_weights(weights)

    def _post_load(self, device):
        raise NotImplementedError('OffPolicyDeepAC is an abstract class. Subclasses need to implement the '
                                  '`_post_load` method.')

    @staticmethod
    def _init_target(online, target):
        target.set_weights(online.get_weights())
