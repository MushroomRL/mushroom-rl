import torch

from copy import deepcopy

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OnPolicyDeepAC
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.rl_utils.value_functions import compute_gae
from mushroom_rl.rl_utils.parameters import Parameter


class TRPO(OnPolicyDeepAC):
    """
    Trust Region Policy optimization algorithm.
    "Trust Region Policy Optimization".
    Schulman J. et al. 2015.

    """

    def __init__(self, mdp_info, policy, critic_params, ent_coeff=0., max_kl=.001, lam=1.,
                 n_epochs_line_search=10, n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10,
                 critic_fit_params=None, backend='torch', history_length=1, action_history_length=0):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            critic_params (dict): parameters of the critic approximator to build;
            ent_coeff ([float, Parameter], 0): coefficient for the entropy penalty;
            max_kl ([float, Parameter], .001): maximum kl allowed for every policy update;
            lam float([float, Parameter], 1.): lambda coefficient used by generalized advantage estimation;
            n_epochs_line_search ([int, Parameter], 10): maximum number of iterations of the line search algorithm;
            n_epochs_cg ([int, Parameter], 10): maximum number of iterations of the conjugate gradient algorithm;
            cg_damping ([float, Parameter], 1e-2): damping factor for the conjugate gradient algorithm;
            cg_residual_tol ([float, Parameter], 1e-10): conjugate gradient residual tolerance;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator;
            history_length (int, 1): number of consecutive observations stacked as policy input;
            action_history_length (int, 0): number of previous actions fed to the actor and critic.

        """
        self._critic_fit_params = dict(n_epochs=5) if critic_fit_params is None else critic_fit_params

        self._n_epochs_line_search = Parameter.make(n_epochs_line_search, backend='torch')
        self._n_epochs_cg = Parameter.make(n_epochs_cg, backend='torch')
        self._cg_damping = Parameter.make(cg_damping, backend='torch')
        self._cg_residual_tol = Parameter.make(cg_residual_tol, backend='torch')

        self._max_kl = Parameter.make(max_kl, backend='torch')
        self._ent_coeff = Parameter.make(ent_coeff, backend='torch')

        self._lambda = Parameter.make(lam, backend='torch')

        self._V = TorchApproximator(**critic_params)

        self._old_policy = None

        super().__init__(mdp_info, policy, backend=backend, history_length=history_length,
                         action_history_length=action_history_length)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_line_search='mushroom',
            _n_epochs_cg='mushroom',
            _cg_damping='mushroom',
            _cg_residual_tol='mushroom',
            _max_kl='mushroom',
            _ent_coeff='mushroom',
            _lambda='mushroom',
            _V='mushroom',
            _old_policy='mushroom'
        )
        self._add_logger_attr('_V', group='critic')

    def fit(self, dataset):
        self._log_iteration_start()

        state_old = self._history_manager.parse_state(dataset)
        self._history_manager.update_preprocessors(dataset)

        state, action, reward, next_state, absorbing, last, extra = self._history_manager.parse_history(dataset)
        prev_action = extra.get('action_history')

        v_target, adv = compute_gae(self._V, state, next_state, reward, absorbing, last,
                                    self.mdp_info.gamma, self._lambda(), action_history=prev_action, action=action)
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        adv = adv.detach()
        v_target = v_target.detach()

        # Policy update
        self._old_policy = deepcopy(self.policy)
        old_pol_dist = self._old_policy.distribution(state_old, action_history=prev_action)
        old_log_prob = self._old_policy.log_prob(state_old, action, action_history=prev_action).detach()

        TorchUtils.zero_grad(self.policy.parameters())
        loss = self._compute_loss(state, action, adv, old_log_prob, prev_action)

        prev_loss = loss.item()

        # Compute Gradient
        loss.backward()
        g = TorchUtils.get_gradient(self.policy.parameters())

        # Compute direction through conjugate gradient
        stepdir = self._conjugate_gradient(g, state, old_pol_dist, prev_action)

        # Line search
        self._line_search(state, action, adv, old_log_prob, old_pol_dist, prev_loss, stepdir, prev_action)

        # VF update
        self._V.fit(state, v_target, action_history=prev_action, **self._critic_fit_params)

        self._log_info(dataset, state, old_pol_dist, action_history=prev_action)

    def _fisher_vector_product(self, p, obs, old_pol_dist, action_history=None):
        kl = self._compute_kl(obs, old_pol_dist, action_history)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = torch.sum(flat_grad_kl * p)
        grads_v = torch.autograd.grad(kl_v, self.policy.parameters(), create_graph=False)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_v]).data

        return flat_grad_grad_kl + p * self._cg_damping()

    def _conjugate_gradient(self, b, obs, old_pol_dist, action_history=None):
        p = b.detach()
        r = b.detach()
        x = torch.zeros_like(p)
        r2 = r.dot(r)

        for i in range(self._n_epochs_cg()):
            z = self._fisher_vector_product(p, obs, old_pol_dist, action_history).detach()
            v = r2 / p.dot(z)
            x += v * p
            r -= v * z
            r2_new = r.dot(r)
            mu = r2_new / r2
            p = r + mu * p

            r2 = r2_new
            if r2 < self._cg_residual_tol():
                break
        return x

    def _line_search(self, obs, act, adv, old_log_prob, old_pol_dist, prev_loss, stepdir, action_history=None):
        # Compute optimal step size
        direction = self._fisher_vector_product(stepdir, obs, old_pol_dist, action_history).detach()
        shs = .5 * stepdir.dot(direction)
        lm = torch.sqrt(shs / self._max_kl())
        full_step = (stepdir / lm).detach()
        stepsize = 1.

        # Save old policy parameters
        theta_old = self.policy.get_weights()

        # Perform Line search
        violation = True

        for _ in range(self._n_epochs_line_search()):
            theta_new = theta_old + full_step * stepsize
            self.policy.set_weights(theta_new)

            new_loss = self._compute_loss(obs, act, adv, old_log_prob, action_history)
            kl = self._compute_kl(obs, old_pol_dist, action_history)
            improve = new_loss - prev_loss
            if kl <= self._max_kl.get_value() * 1.5 and improve >= 0:
                violation = False
                break
            stepsize *= .5

        if violation:
            self.policy.set_weights(theta_old)

    def _compute_kl(self, obs, old_pol_dist, action_history=None):
        new_pol_dist = self.policy.distribution(obs, action_history=action_history)
        return torch.mean(torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist))

    def _compute_loss(self, obs, act, adv, old_log_prob, action_history=None):
        ratio = torch.exp(self.policy.log_prob(obs, act, action_history=action_history) - old_log_prob)
        J = torch.mean(ratio * adv)

        return J + self._ent_coeff() * self.policy.entropy(obs)
