import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.torch import get_gradient, zero_grad
from mushroom.utils.minibatches import minibatch_generator
from mushroom.utils.dataset import parse_dataset, compute_J


def compute_gae(V, s, ss, r, absorbing, last, gamma, lam):
    v = V(s)
    v_next = V(ss)
    gen_adv = np.empty_like(v)
    for rev_k, _ in enumerate(reversed(v)):
        k = len(v) - rev_k - 1
        if last[k] or rev_k == 0:
            gen_adv[k] = r[k] - v[k]
            if not absorbing[k]:
                gen_adv[k] += gamma * v_next[k]
        else:
            gen_adv[k] = r[k] + gamma * v_next[k] - v[k] + gamma * lam * gen_adv[k + 1]
    return gen_adv + v, gen_adv


class TRPO(Agent):
    def __init__(self, mdp_info, policy, critic_params,
                 ent_coeff=0., max_kl=.001, lam=1.,
                 n_epochs_v=3, n_epochs_line_search=10, n_epochs_cg=10,
                 cg_damping=1e-2, cg_residual_tol=1e-10, quiet=True):
        """
        Constructor.

        Args:


        """
        self._n_epochs_line_search = n_epochs_line_search
        self._n_epochs_v = n_epochs_v
        self._n_epochs_cg = n_epochs_cg
        self._cg_damping = cg_damping
        self._cg_residual_tol = cg_residual_tol

        self._max_kl = max_kl
        self._ent_coeff = ent_coeff

        self._lambda = lam

        self._V = Regressor(TorchApproximator, **critic_params)

        self._iter = 1
        self._quiet = quiet

        super().__init__(policy, mdp_info, None)

    def fit(self, dataset):
        if not self._quiet:
            tqdm.write('Iteration ' + str(self._iter))

        state, action, reward, next_state, absorbing, last = parse_dataset(dataset)
        x = state.astype(np.float32)
        u = action.astype(np.float32)
        r = reward.astype(np.float32)
        xn = next_state.astype(np.float32)

        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last,
                                       self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = torch.tensor(np_adv, dtype=torch.float)

        # Policy update
        old_pol_dist = self.policy.distribution_t(obs)
        old_log_prob = self.policy.log_prob_t(obs, act).detach()

        self._zero_grad()
        loss = self._compute_loss(obs, act, adv, old_log_prob)

        prev_loss = loss.item()

        # Compute Gradient
        loss.backward(retain_graph=True)
        g = get_gradient(self.policy.parameters())

        # Compute direction trough conjugate gradient
        stepdir = self._conjugate_gradient(g, obs, old_pol_dist)

        # Line search
        shs = .5 * stepdir.dot(self._fisher_vector_product(
            torch.from_numpy(stepdir), obs, old_pol_dist)
        )
        lm = np.sqrt(shs / self._max_kl)
        fullstep = stepdir / lm
        stepsize = 1.

        theta_old = self.policy.get_weights()

        violation = True

        for _ in range(self._n_epochs_line_search):
            theta_new = theta_old + fullstep * stepsize
            self.policy.set_weights(theta_new)

            new_loss = self._compute_loss(obs, act, adv, old_log_prob)
            kl = self._compute_kl(obs, old_pol_dist)
            improve = new_loss - prev_loss
            if kl <= self._max_kl * 1.5 or improve >= 0:
                violation = False
                break
            stepsize *= .5

        if violation:
            self.policy.set_weights(theta_old)

        # VF update
        self._V.fit(x, v_target, n_epochs=self._n_epochs_v)

        # Print fit information
        self._print_fit_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _zero_grad(self):
        zero_grad(self.policy.parameters())

    def _conjugate_gradient(self, b, obs, old_pol_dist):
        p = b.detach().numpy()
        r = b.detach().numpy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)

        for i in range(self._n_epochs_cg):
            z = self._fisher_vector_product(
                torch.from_numpy(p), obs, old_pol_dist).detach().numpy()
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < self._cg_residual_tol:
                break
        return x

    def _fisher_vector_product(self, p, obs, old_pol_dist):
        self._zero_grad()
        kl = self._compute_kl(obs, old_pol_dist)
        grads = torch.autograd.grad(kl, self.policy.parameters(),
                                    create_graph=True, retain_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * torch.autograd.Variable(p)).sum()
        grads = torch.autograd.grad(kl_v, self.policy.parameters(),
                                    retain_graph=True)
        flat_grad_grad_kl = torch.cat(
            [grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + p * self._cg_damping

    def _compute_kl(self, obs, old_pol_dist):
        new_pol_dist = self.policy.distribution_t(obs)
        return torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist,
                                                               old_pol_dist))

    def _compute_loss(self, obs, act, adv, old_log_prob):
        ratio = torch.exp(self.policy.log_prob_t(obs, act) - old_log_prob)
        J = torch.mean(ratio * adv)

        return J + self._ent_coeff * self.policy.entropy_t(obs)

    def _print_fit_info(self, dataset, x, v_target, old_pol_dist):
        if not self._quiet:
            logging_verr = []
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            for idx in range(len(self._V)):
                v_pred = torch.tensor(self._V(x, idx=idx), dtype=torch.float)
                v_err = F.mse_loss(v_pred, torch_v_targets)
                logging_verr.append(v_err.item())

            logging_ent = self.policy.entropy(x)
            new_pol_dist = self.policy.distribution(x)
            logging_kl = torch.mean(
                torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist)
            )
            avg_rwd = np.mean(compute_J(dataset))
            tqdm.write("Iterations Results:\n\trewards {} vf_loss {}\n\tentropy {}  kl {}".format(
                avg_rwd, logging_verr, logging_ent, logging_kl))
            tqdm.write(
                '--------------------------------------------------------------------------------------------------')


class PPO(Agent):
    def __init__(self, mdp_info, policy, critic_params, lr_p, n_epochs_v,
                 n_epochs_policy, batch_size, eps_ppo, lam, quiet=True):
        self._n_epochs_policy = n_epochs_policy
        self._n_epochs_v = n_epochs_v
        self._batch_size = batch_size
        self._eps_ppo = eps_ppo

        self._p_optim = optim.Adam(policy.parameters(), lr=lr_p)

        self._lambda = lam

        self._V = Regressor(TorchApproximator, **critic_params)

        self._quiet = quiet
        self._iter = 1

        super().__init__(policy, mdp_info, None)

    def fit(self, dataset):
        if not self._quiet:
            tqdm.write('Iteration ' + str(self._iter))

        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = torch.tensor(np_adv, dtype=torch.float)

        old_pol_dist = self.policy.distribution_t(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x, v_target, n_epochs=self._n_epochs_v)

        self._update_policy(obs, act, adv, old_log_p)

        # Print fit information
        self._print_fit_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size, obs, act, adv, old_log_p):
                self._p_optim.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i, act_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo,
                                            1 + self._eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss.backward()
                self._p_optim.step()

    def _print_fit_info(self, dataset, x, v_target, old_pol_dist):
        if not self._quiet:
            logging_verr = []
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            for idx in range(len(self._V)):
                v_pred = torch.tensor(self._V(x, idx=idx), dtype=torch.float)
                v_err = F.mse_loss(v_pred, torch_v_targets)
                logging_verr.append(v_err.item())

            logging_ent = self.policy.entropy(x)
            new_pol_dist = self.policy.distribution(x)
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(
                new_pol_dist, old_pol_dist))
            avg_rwd = np.mean(compute_J(dataset))
            tqdm.write("Iterations Results:\n\trewards {} vf_loss {}\n\tentropy {}  kl {}".format(
                avg_rwd, logging_verr, logging_ent, logging_kl))
            tqdm.write(
                '--------------------------------------------------------------------------------------------------')
