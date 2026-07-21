import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OnPolicyDeepAC
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.value_functions import compute_gae
from mushroom_rl.rl_utils.parameters import Parameter


class PPO(OnPolicyDeepAC):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al. 2017.

    """

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, n_epochs_policy, batch_size, eps_ppo, lam,
                 ent_coeff=0.0, critic_fit_params=None):
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

        self._n_epochs_policy = Parameter.make(n_epochs_policy, backend='torch')
        self._batch_size = Parameter.make(batch_size, backend='torch')
        self._eps_ppo = Parameter.make(eps_ppo, backend='torch')

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._lambda = Parameter.make(lam, backend='torch')
        self._ent_coeff = Parameter.make(ent_coeff, backend='torch')

        self._V = TorchApproximator(**critic_params)

        super().__init__(mdp_info, policy, backend='torch')

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom'
        )
        self._add_logger_attr('_V', group='critic')

    def fit(self, dataset):
        self._log_iteration_start()

        state, action, reward, next_state, absorbing, last = dataset.parse(to='torch')
        state, next_state, state_old = self._preprocess_state(state, next_state)

        v_target, adv = compute_gae(self._V, state, next_state, reward, absorbing, last, self.mdp_info.gamma,
                                    self._lambda())
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        adv = adv.detach()
        v_target = v_target.detach()

        old_pol_dist = self.policy.distribution(state_old)
        old_log_p = old_pol_dist.log_prob(action)[:, None].detach()

        self._V.fit(state, v_target, **self._critic_fit_params)

        self._update_policy(state, action, adv, old_log_p)

        self._log_info(dataset, state, old_pol_dist)

    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(self._batch_size(), obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(self.policy.log_prob(obs_i, act_i) - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss -= self._ent_coeff() * self.policy.entropy(obs_i)
                loss.backward()
                self._optimizer.step()

    def _post_load(self):
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))
