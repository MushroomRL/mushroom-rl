import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import parse_dataset, compute_J
from mushroom_rl.utils.value_functions import compute_gae


class PPO(Agent):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, quiet=True,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            n_epochs_policy (int): number of policy updates for every dataset;
            batch_size (int): size of minibatches for every optimization step
            eps_ppo (float): value for probability ratio clipping;
            lam float(float, 1.): lambda coefficient used by generalized
                advantage estimation;
            quiet (bool, True): if true, the algorithm will print debug
                information;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict(n_epochs=10) if critic_fit_params is None else critic_fit_params

        self._n_epochs_policy = n_epochs_policy
        self._batch_size = batch_size
        self._eps_ppo = eps_ppo

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._lambda = lam

        self._V = Regressor(TorchApproximator, **critic_params)

        self._quiet = quiet
        self._iter = 1

        self._add_save_attr(
            _critic_fit_params='pickle', 
            _n_epochs_policy='primitive',
            _batch_size='primitive',
            _eps_ppo='primitive',
            _optimizer='torch',
            _lambda='primitive',
            _V='mushroom',
            _quiet='primitive',
            _iter='primitive'
        )

        super().__init__(mdp_info, policy, None)

    def fit(self, dataset):
        if not self._quiet:
            tqdm.write('Iteration ' + str(self._iter))

        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)
        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x, v_target, **self._critic_fit_params)

        self._update_policy(obs, act, adv, old_log_p)

        # Print fit information
        self._print_fit_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size, obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i, act_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo,
                                            1 + self._eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss.backward()
                self._optimizer.step()

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
