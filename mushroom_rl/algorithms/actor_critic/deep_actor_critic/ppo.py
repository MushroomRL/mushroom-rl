import numpy as np

import torch
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OnPolicyDeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.value_functions import compute_gae
from mushroom_rl.rl_utils.parameters import to_parameter


class PPO(OnPolicyDeepAC):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            n_epochs_policy ([int, Parameter]): number of policy updates for every dataset;
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            eps_ppo ([float, Parameter]): value for probability ratio clipping;
            lam ([float, Parameter], 1.): lambda coefficient used by generalized
                advantage estimation;
            ent_coeff ([float, Parameter], 1.): coefficient for the entropy regularization term;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict(n_epochs=10) if critic_fit_params is None else critic_fit_params

        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._batch_size = to_parameter(batch_size)
        self._eps_ppo = to_parameter(eps_ppo)

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._lambda = to_parameter(lam)
        self._ent_coeff = to_parameter(ent_coeff)

        self._V = Regressor(TorchApproximator, **critic_params)

        self._iter = 1

        super().__init__(mdp_info, policy, backend='torch')

        self._add_save_attr(
            _critic_fit_params='pickle', 
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom',
            _iter='primitive'
        )

    def fit(self, dataset):
        state, action, reward, next_state, absorbing, last = dataset.parse(to='torch')
        state, next_state, state_old = self._preprocess_state(state, next_state)

        v_target, adv = compute_gae(self._V, state, next_state, reward, absorbing, last, self.mdp_info.gamma,
                                    self._lambda())
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        adv = adv.detach()
        v_target = v_target.detach()

        old_pol_dist = self.policy.distribution_t(state_old)
        old_log_p = old_pol_dist.log_prob(action)[:, None].detach()

        self._V.fit(state, v_target, **self._critic_fit_params)

        self._update_policy(state, action, adv, old_log_p)

        # Print fit information
        self._log_info(dataset, state, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(self.policy.log_prob_t(obs_i, act_i) - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss -= self._ent_coeff() * self.policy.entropy_t(obs_i)
                loss.backward()
                self._optimizer.step()

    def _log_info(self, dataset, x, v_target, old_pol_dist):
        if self._logger:
            with torch.no_grad():
                logging_verr = []
                for idx in range(len(self._V)):
                    v_pred = self._V(x, idx=idx)
                    v_err = F.mse_loss(v_pred, v_target)
                    logging_verr.append(v_err.item())

                logging_ent = self.policy.entropy(x)
                new_pol_dist = self.policy.distribution(x)
                logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
                avg_rwd = np.mean(dataset.undiscounted_return)
                msg = "Iteration {}:\n\t\t\t\trewards {} vf_loss {}\n\t\t\t\tentropy {}  kl {}".format(
                    self._iter, avg_rwd, logging_verr, logging_ent, logging_kl)

                self._logger.info(msg)
                self._logger.weak_line()

    def _post_load(self):
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))
