from mushroom_rl.core import Agent
from mushroom_rl.core.asymmetric_history_manager import AsymmetricHistoryManager
from mushroom_rl.utils.torch_utils import TorchUtils


class DeepAC(Agent):
    """
    Base class for deep actor-critic algorithms.

    """
    _asymmetric_options = None

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
        self._parameters = None
        self._optimizer = None
        self._clipping = None
        self._clipping_params = None

        if actor_optimizer is not None:
            if parameters is not None and not isinstance(parameters, list):
                parameters = list(parameters)
            self._parameters = parameters

            self._optimizer = actor_optimizer['class'](parameters, **actor_optimizer['params'])

            if 'clipping' in actor_optimizer:
                self._clipping = actor_optimizer['clipping']['method']
                self._clipping_params = actor_optimizer['clipping']['params']

        asymmetric_options = self._asymmetric_options or dict()
        history_params = dict(history_length=history_length, action_history_length=action_history_length)
        history_params.update(asymmetric_options)

        super().__init__(mdp_info, policy, backend=backend, device=device, **history_params)

        self._check_device(policy, 'policy')

        self._add_save_attr(
            _parameters='none',
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

    @classmethod
    def asymmetric(cls, *args, actor_indices=None, critic_indices=None, critic_history_length=1, **kwargs):
        """
        Build the algorithm as an asymmetric actor-critic, whose actor and critic read different entries of the
        observation, each with its own history length. Both read the observation after the agent preprocessors. The
        actor reads ``actor_indices``, stacked over the ``history_length`` of the algorithm, online and when fitting;
        the critic reads ``critic_indices``, stacked over ``critic_history_length``, when fitting. The networks must be
        built for these inputs: the actor for ``(history_length, len(actor_indices))``, the critic for
        ``(critic_history_length, len(critic_indices))``, without the leading axis when the length is 1.

        Args:
            *args: the positional parameters of the algorithm constructor;
            actor_indices (list, None): the entries of the observation the actor reads; all of them when ``None``;
            critic_indices (list, None): the entries of the observation the critic reads; all of them when ``None``;
            critic_history_length (int, 1): number of observations stacked as critic input;
            **kwargs: the keyword parameters of the algorithm constructor.

        Returns:
            The algorithm instance.

        """
        agent = cls.__new__(cls)
        agent._asymmetric_options = dict(actor_indices=actor_indices, critic_indices=critic_indices,
                                         critic_history_length=critic_history_length)
        agent.__init__(*args, **kwargs)
        del agent._asymmetric_options

        return agent

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

    def _build_history_manager(self, history_length=None, action_history_length=None, **asymmetric_params):
        if asymmetric_params:
            return AsymmetricHistoryManager.default_streams(self.mdp_info, self._info, history_length,
                                                            action_history_length, **asymmetric_params)
        else:
            return super()._build_history_manager(history_length=history_length,
                                                  action_history_length=action_history_length)

    def _post_load(self, device):
        if self._optimizer is not None:
            self.overwrite_actor_parameters(self.policy.parameters())


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
        Open the log block of the current iteration. To be called at the beginning of the fit.

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
    def __init__(self, mdp_info, policy, actor_optimizer, parameters, **kwargs):
        """
        Constructor.

        Args:
            actor_optimizer (dict): parameters to specify the actor optimizer algorithm;
            parameters (list): policy parameters to be optimized by the actor optimizer;
            **kwargs: parameters forwarded to the ``DeepAC`` constructor.

        """
        super().__init__(mdp_info, policy, actor_optimizer, parameters, **kwargs)

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
        super()._post_load(device)
        self._replay_memory._history_manager = self._history_manager

    @staticmethod
    def _init_target(online, target):
        target.set_weights(online.get_weights())
