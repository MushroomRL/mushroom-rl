from mushroom_rl.core import Agent
from mushroom_rl.utils.torch import update_optimizer_parameters


class DeepAC(Agent):
    """
    Base class for algorithms that uses the reparametrization trick, such as
    SAC, DDPG and TD3.

    """

    def __init__(self, mdp_info, policy, actor_optimizer, parameters):
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

            self._optimizer = actor_optimizer['class'](
                parameters, **actor_optimizer['params']
            )

            self._clipping = None

            if 'clipping' in actor_optimizer:
                self._clipping = actor_optimizer['clipping']['method']
                self._clipping_params = actor_optimizer['clipping']['params']
        
        self._add_save_attr(
            _optimizer='torch',
            _clipping='torch',
            _clipping_params='pickle'
        )

        super().__init__(mdp_info, policy)

    def fit(self, dataset, **info):
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
            update_optimizer_parameters(self._optimizer, self._parameters)

    def _post_load(self):
        raise NotImplementedError('DeepAC is an abstract class. Subclasses need'
                                  'to implement the `_post_load` method.')
