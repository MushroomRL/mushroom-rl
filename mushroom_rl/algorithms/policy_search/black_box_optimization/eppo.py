import torch

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import to_parameter


class ePPO(BlackBoxOptimization):
    """
    Episodic adaptation of the Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, distribution, policy, optimizer, n_epochs_policy, batch_size, eps_ppo, ent_coeff=0.0,
                 context_builder=None):
        """
        Constructor.

        Args:
            optimizer: the gradient step optimizer.

        """
        assert hasattr(distribution, 'parameters')

        self._optimizer = optimizer['class'](distribution.parameters(), **optimizer['params'])
        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._batch_size = to_parameter(batch_size)
        self._eps_ppo = to_parameter(eps_ppo)
        self._ent_coeff = to_parameter(ent_coeff)

        super().__init__(mdp_info, distribution, policy, context_builder=context_builder, backend='torch')

        self._add_save_attr(
            _optimizer='torch',
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
        )

    def _update(self, Jep, theta, context):
        Jep = torch.tensor(Jep)
        J_mean = torch.mean(Jep)
        J_std = torch.std(Jep)

        Jep = (Jep - J_mean) / (J_std + 1e-8)

        old_dist = self.distribution.log_pdf(theta).detach()

        if self.distribution.is_contextual:
            full_batch = (theta, Jep, old_dist, context)
        else:
            full_batch = (theta, Jep, old_dist)

        for epoch in range(self._n_epochs_policy()):
            for minibatch in minibatch_generator(self._batch_size(), *full_batch):

                theta_i, context_i, Jep_i, old_dist_i = self._unpack(minibatch)

                self._optimizer.zero_grad()
                prob_ratio = torch.exp(self.distribution.log_pdf(theta_i, context_i) - old_dist_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * Jep_i, clipped_ratio * Jep_i))
                loss -= self._ent_coeff() * self.distribution.entropy(context_i)
                loss.backward()
                self._optimizer.step()

    def _unpack(self, minibatch):
        if self.distribution.is_contextual:
            theta_i, Jep_i, old_dist_i, context_i = minibatch
        else:
            theta_i, Jep_i, old_dist_i = minibatch
            context_i = None

        return theta_i, context_i, Jep_i, old_dist_i
