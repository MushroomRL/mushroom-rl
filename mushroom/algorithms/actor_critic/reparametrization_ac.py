from mushroom.algorithms.agent import Agent


class ReparametrizationAC(Agent):
    """
    Base class for algorithms that uses the reparametrization trick, such as
    SAC, DDPG and TD3
    """
    def __init__(self, policy, mdp_info, optimizer, parameters):
        if optimizer is not None:
            self._optimizer = optimizer['class'](parameters,
                                                 **optimizer['params'])

        super().__init__(policy, mdp_info)

    def _optimize_actor_parameters(self, loss):
        """
        Method used to update actor parameters to maximize a given loss
        Args:
            loss (torch.tensor): the loss computed by the algorithm;
        """
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
