import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import PPO
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.value_functions import compute_gae

class RudinPPO(PPO):
    """
    Extended PPO algorithm as "Learning to walk in minutes using massively parallel deep reinforcement learning" paper.
    Introducing gradinet clipping and adaptive leanring rate based on KL divergence.
    """

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
                 critic_fit_params=None, clip_grad_norm=1., schedule='adaptive', desired_kl=0.01):
        super().__init__(mdp_info, policy, actor_optimizer, critic_params,
                            n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff,
                            critic_fit_params)

        self._clip_grad_norm = clip_grad_norm
        self._schedule = schedule
        self._desired_kl = desired_kl
        self._actor_learning_rate = actor_optimizer['params']['lr']
        self._critic_learning_rate = critic_params['optimizer']['params']['lr']

        self._add_save_attr(
            _clip_grad_norm='primitive',
            _schedule='primitive',
            _desired_kl='primitive',
            _actor_learning_rate='primitive',
            _critic_learning_rate='primitive',
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

        self._update_policy(state, action, adv, old_log_p, state, old_pol_dist)

        # Print fit information
        self._log_info(dataset, state, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, obs, act, adv, old_log_p, state, old_pol_dist):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p):
                with torch.inference_mode():
                    new_pol_dist = self.policy.distribution_t(state)
                    kl = torch.mean(torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist))
                    self._adapt_learning_rate(kl)

                self._optimizer.zero_grad()
                prob_ratio = torch.exp(self.policy.log_prob_t(obs_i, act_i) - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss -= self._ent_coeff() * self.policy.entropy_t(obs_i)
                loss.backward()
                self._clip_gradient()
                self._optimizer.step()

    def _clip_gradient(self):
        if self._clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._clip_grad_norm)

    def _adapt_learning_rate(self, kl):
        if self._schedule == 'adaptive' and self._desired_kl:
            if kl > 2.0 * self._desired_kl:
                self._actor_learning_rate = max(1e-5, self._actor_learning_rate / 1.5)
                self._critic_learning_rate = max(1e-5, self._critic_learning_rate / 1.5)
            elif kl < 0.5 * self._desired_kl and kl > 0.0:
                self._actor_learning_rate = min(1e-2, self._actor_learning_rate * 1.5)
                self._critic_learning_rate = min(1e-2, self._critic_learning_rate * 1.5)

            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self._actor_learning_rate

            for param_group in self._V._impl.model._optimizer.param_groups:
                param_group['lr'] = self._critic_learning_rate
