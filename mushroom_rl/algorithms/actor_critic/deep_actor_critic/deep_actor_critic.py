import torch

from mushroom_rl.core import Agent
from mushroom_rl.utils.torch_utils import TorchUtils


class OnPolicyDeepAC(Agent):
    """
    Base class for on policy deep actor-critic algorithms, such as PPO and TRPO.

    """
    def __init__(self, mdp_info, policy, **kwargs):
        """
        Constructor.

        Args:
            **kwargs: parameters forwarded to the ``Agent`` constructor.

        """
        self._iter = 1

        super().__init__(mdp_info, policy, **kwargs)

        self._add_save_attr(_iter='primitive')

    def _preprocess_state(self, state, next_state, output_old=True):
        state_old = None

        if output_old:
            state_old = self._agent_preprocess(state)

        self._update_agent_preprocessor(state)
        state = self._agent_preprocess(state)
        next_state = self._agent_preprocess(next_state)

        if output_old:
            return state, next_state, state_old
        else:
            return state, next_state

    def _log_iteration_start(self):
        """
        Open the log block of the current iteration. To be called at the beginning of the fit, so that
        the quantities logged during the fit, e.g. the critic loss, appear under the iteration number.

        """
        if self._logger:
            self._logger.weak_line(debug=True)
            self._logger.debug('Iteration ' + str(self._iter))

    def _log_info(self, dataset, x, old_pol_dist, *args, **kwargs):
        """
        Log the fit information of the current iteration and advance the iteration counter. The extra
        arguments are forwarded to the policy distribution, to support policies requiring more than the
        state to be evaluated.

        Args:
            dataset (Dataset): the dataset used by the current fit;
            x (torch.Tensor): the states used to evaluate the policy;
            old_pol_dist (torch.distributions.Distribution): the policy distribution before the fit;
            *args: additional positional arguments for the policy distribution;
            **kwargs: additional keyword arguments for the policy distribution.

        """
        if self._logger:
            with torch.no_grad():
                new_pol_dist = self.policy.distribution(x, *args, **kwargs)
                kl = torch.mean(torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist))

                self._logger.log_training(R=dataset.undiscounted_return.mean().item())
                self._logger.log_training('actor', entropy=self.policy.entropy(x).item(), kl=kl.item())
                self._logger.advance_step()

        self._iter += 1


class DeepAC(Agent):
    """
    Base class for off policy deep actor-critic algorithms.
    These algorithms use the reparametrization trick, such as SAC, DDPG and TD3.

    """

    def __init__(self, mdp_info, policy, actor_optimizer, parameters, backend='torch'):
        """
        Constructor.

        Args:
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            parameters (list): policy parameters to be optimized.

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

        super().__init__(mdp_info, policy, backend=backend)

        self._add_save_attr(
            _optimizer='torch',
            _clipping='torch',
            _clipping_params='pickle'
        )

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError('DeepAC is an abstract class')

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

    @staticmethod
    def _init_target(online, target):
        for i in range(len(target)):
            target[i].set_weights(online[i].get_weights())

    def _update_target(self, online, target):
        for i in range(len(target)):
            weights = self._tau() * online[i].get_weights()
            weights += (1 - self._tau.get_value()) * target[i].get_weights()
            target[i].set_weights(weights)

    def _update_optimizer_parameters(self, parameters):
        self._parameters = list(parameters)
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, self._parameters)

    def _post_load(self):
        raise NotImplementedError('DeepAC is an abstract class. Subclasses need to implement the `_post_load` method.')
