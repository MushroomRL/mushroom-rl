import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OnPolicyDeepAC
from mushroom_rl.core import SequenceBatch
from mushroom_rl.approximators.parametric import RecurrentTorchApproximator
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import Parameter


class PPO_BPTT(OnPolicyDeepAC):
    """
    Backpropagation trough time extension of the  Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, n_epochs_policy, batch_size, eps_ppo, lam,
                 dim_env_state, ent_coeff=0.0, critic_fit_params=None, truncation_length=5, history_length=1,
                 action_history_length=0, device=None):
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
                of the critic approximator;
            truncation_length (int, 5): truncation length of the backpropagation through time;
            history_length (int, 1): number of observations stacked as input at each timestep; when greater than 1 the
                agent stacks the observation history online and the same window is rebuilt for each sequence timestep
                during the fit, so each sequence entry becomes a ``(history_length, *obs_shape)`` stack;
            action_history_length (int, 0): number of previous actions fed to the policy at each timestep; when
                greater than 0 the agent assembles the previous-action window online and the same window is rebuilt
                for each sequence timestep during the fit;
            device (str, None): device the agent data is stored on, or ``None`` for the default one.

        """
        self._critic_fit_params = dict(n_epochs=10) if critic_fit_params is None else critic_fit_params

        self._n_epochs_policy = Parameter.make(n_epochs_policy, backend='torch', device=device)
        self._batch_size = Parameter.make(batch_size, backend='torch', device=device)
        self._eps_ppo = Parameter.make(eps_ppo, backend='torch', device=device)

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._lambda = Parameter.make(lam, backend='torch', device=device)
        self._ent_coeff = Parameter.make(ent_coeff, backend='torch', device=device)

        if device is not None:
            critic_params.setdefault('device', device)
        self._V = RecurrentTorchApproximator(**critic_params)

        self._truncation_length = truncation_length
        self._dim_env_state = dim_env_state

        super().__init__(mdp_info, policy, backend='torch', device=device, history_length=history_length,
                         action_history_length=action_history_length)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom',
            _dim_env_state='primitive',
            _truncation_length='primitive'
        )
        self._add_logger_attr('_V', group='critic')

    def fit(self, dataset):
        self._log_iteration_start()

        state_old = self._history_manager.parse_state(dataset)
        self._history_manager.update_preprocessors(dataset)

        batch = self._history_manager.parse_history(dataset, get_policy_state=True)
        sequences, state_old_seq = self._transform_to_sequences(batch, state_old)

        v_target, adv = self.compute_gae(self._V, sequences, self.mdp_info.gamma, self._lambda())
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        with torch.no_grad():
            old_pol_dist = self.policy.distribution(state_old_seq, sequences.policy_state, sequences.lengths,
                                                    action_history=sequences.action_history)

        old_log_p = old_pol_dist.log_prob(batch.action).detach()

        self._V.fit(sequences.state, sequences.policy_state, sequences.lengths, v_target,
                    action_history=sequences.action_history, **self._critic_fit_params)

        kl, entropy = self._update_policy(sequences.state, sequences.policy_state, batch.action, sequences.lengths, adv,
                                          old_log_p, sequences.action_history)

        self._log_info(dataset, kl, entropy)

    @staticmethod
    def compute_gae(V, sequences, gamma, lam):
        """
        Function to compute Generalized Advantage Estimation (GAE)
        and new value function target over a dataset.

        "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation".
        Schulman J. et al.. 2016.

        Args:
            V (Approximator): the current value function approximator;
            sequences (SequenceBatch): the sequences ending at each step of whole trajectories, in the order of the
                steps;
            gamma (float): the discount factor of the considered problem;
            lam (float): the value for the lamba coefficient used by GEA
                algorithm.
        Returns:
            The new estimate for the value function of the next state
            and the estimated generalized advantage.
        """
        with torch.no_grad():
            v = V(sequences.state, sequences.policy_state, lengths=sequences.lengths,
                  action_history=sequences.action_history)
            v_next = V(sequences.next_state, sequences.next_policy_state, lengths=sequences.lengths,
                       action_history=sequences.next_action_history)
            r, absorbing, last = sequences.reward, sequences.absorbing, sequences.last
            gen_adv = torch.empty_like(v)
            for rev_k in range(len(v)):
                k = len(v) - rev_k - 1
                if last[k] or rev_k == 0:
                    gen_adv[k] = r[k] - v[k]
                    if not absorbing[k]:
                        gen_adv[k] += gamma * v_next[k]
                else:
                    gen_adv[k] = r[k] + gamma * v_next[k] - v[k] + gamma * lam * gen_adv[k + 1]

            return gen_adv + v, gen_adv

    def _transform_to_sequences(self, batch, states_old):
        n_steps = len(batch.state)

        with torch.no_grad():
            steps = torch.arange(n_steps, device=batch.last.device)
            boundary = (batch.last.int() + batch.absorbing.int()) > 0
            previous_boundary = torch.cummax(torch.where(boundary, steps, -1), dim=0).values
            previous_boundary = torch.cat([previous_boundary.new_full((1,), -1), previous_boundary[:-1]])
            begin = torch.maximum(steps - self._truncation_length + 1, previous_boundary + 1)
            lengths = steps - begin + 1

            offsets = torch.arange(self._truncation_length, device=steps.device)
            rows = torch.clamp(begin[:, None] + offsets, max=n_steps - 1)
            valid = offsets < lengths[:, None]

            state, action, next_state, state_old = self._gather_sequences(rows, valid, batch.state, batch.action,
                                                                          batch.next_state, states_old)
            action_history = next_action_history = None
            if self._history_manager.uses_action:
                action_history, next_action_history = self._gather_sequences(rows, valid, batch.action_history,
                                                                             batch.next_action_history)
            policy_state, next_policy_state = self._gather_sequence_starts(begin, batch.policy_state,
                                                                           batch.next_policy_state)

            sequences = SequenceBatch(state, action, batch.reward, next_state, batch.absorbing, batch.last,
                                      action_history=action_history, next_action_history=next_action_history,
                                      policy_state=policy_state, next_policy_state=next_policy_state,
                                      lengths=lengths.cpu())

            return sequences, state_old

    def _gather_sequences(self, rows, valid, *arrays):
        """
        Gather the sequence of every step from each array, zero-padded after its final step, into the default dtype
        on the agent device.

        """
        sequences = list()
        for array in arrays:
            mask = valid.reshape(valid.shape + (1,) * (array.dim() - 1))
            padded = torch.where(mask, array[rows], 0)
            sequences.append(torch.empty(padded.shape, device=self._info.device).copy_(padded))
        return sequences

    def _gather_sequence_starts(self, begin, *arrays):
        """
        Gather the entry at the first step of the sequence of every step from each array, into the default dtype on
        the agent device.

        """
        starts = list()
        for array in arrays:
            start = array[begin]
            starts.append(torch.empty(start.shape, device=self._info.device).copy_(start))
        return starts

    def _update_policy(self, obs, pi_h, act, lengths, adv, old_log_p, prev_action):
        tensors = (obs, pi_h, act, lengths, adv, old_log_p) if not self._history_manager.uses_action \
            else (obs, pi_h, act, lengths, adv, old_log_p, prev_action)
        kls = list()
        entropies = list()
        for epoch in range(self._n_epochs_policy()):
            for batch in minibatch_generator(self._batch_size(), *tensors):
                obs_i, pi_h_i, act_i, length_i, adv_i, old_log_p_i, *rest = batch
                prev_action_i = rest[0] if rest else None
                self._optimizer.zero_grad()
                log_p = self.policy.log_prob(obs_i, act_i, pi_h_i, length_i, action_history=prev_action_i)
                log_ratio = log_p - old_log_p_i
                prob_ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                entropy = self.policy.entropy(obs_i)
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss -= self._ent_coeff() * entropy
                loss.backward()
                self._optimizer.step()

                kl_est = ((prob_ratio - 1) - log_ratio).detach()
                kls.append(kl_est.mean())
                entropies.append(entropy.detach())

        return torch.stack(kls).mean(), torch.stack(entropies).mean()

    def _post_load(self, device):
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))
