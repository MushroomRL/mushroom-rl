import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OnPolicyDeepAC
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
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, dim_env_state, ent_coeff=0.0,
                 critic_fit_params=None, truncation_length=5, history_length=1, action_history_length=0):
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
                for each sequence timestep during the fit.

        """
        self._critic_fit_params = dict(n_epochs=10) if critic_fit_params is None else critic_fit_params

        self._n_epochs_policy = Parameter.make(n_epochs_policy, backend='torch')
        self._batch_size = Parameter.make(batch_size, backend='torch')
        self._eps_ppo = Parameter.make(eps_ppo, backend='torch')

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._lambda = Parameter.make(lam, backend='torch')
        self._ent_coeff = Parameter.make(ent_coeff, backend='torch')

        self._V = RecurrentTorchApproximator(**critic_params)

        self._truncation_length = truncation_length
        self._dim_env_state = dim_env_state

        super().__init__(mdp_info, policy, backend='torch', history_length=history_length,
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

        state, action, reward, next_state, absorbing, last, extra = self._history_manager.parse_history(dataset)
        state, next_state, state_old = self._preprocess_state(state, next_state)
        prev_action = extra.get('action_history')

        policy_state, policy_next_state = dataset.parse_policy_state(to='torch')
        state_old_seq, state_seq, policy_state_seq, act_seq, state_next_seq, policy_next_state_seq, prev_action_seq, \
            lengths = self._transform_to_sequences(state_old, state, policy_state, action, next_state,
                                                   policy_next_state, prev_action, last, absorbing)

        v_target, adv = self.compute_gae(self._V, state_seq, policy_state_seq, state_next_seq, policy_next_state_seq,
                                         lengths, reward, absorbing, last, self.mdp_info.gamma, self._lambda(),
                                         action_history=prev_action_seq,
                                         next_action_history=act_seq if self._history_manager.uses_action else None)
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        old_pol_dist = self.policy.distribution(state_old_seq, policy_state_seq, lengths,
                                                action_history=prev_action_seq)
        old_log_p = old_pol_dist.log_prob(action)[:, None].detach()

        critic_args = (state_seq, policy_state_seq, lengths)
        critic_args += (prev_action_seq,) if self._history_manager.uses_action else ()
        self._V.fit(*critic_args, v_target, **self._critic_fit_params)

        self._update_policy(state_seq, policy_state_seq, action, lengths, adv, old_log_p, prev_action_seq)

        self._log_info(dataset, state_seq, old_pol_dist, policy_state_seq, lengths, action_history=prev_action_seq)

    @staticmethod
    def compute_gae(V, s, pi_h, ss, pi_hn, lengths, r, absorbing, last, gamma, lam,
                    action_history=None, next_action_history=None):
        """
        Function to compute Generalized Advantage Estimation (GAE)
        and new value function target over a dataset.

        "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation".
        Schulman J. et al.. 2016.

        Args:
            V (Regressor): the current value function regressor;
            s (torch.Tensor): the set of states in which we want
                to evaluate the advantage;
            ss (torch.Tensor): the set of next states in which we want
                to evaluate the advantage;
            r (torch.Tensor): the reward obtained in each transition
                from state s to state ss;
            absorbing (torch.Tensor): an array of boolean flags indicating
                if the reached state is absorbing;
            last (torch.Tensor): an array of boolean flags indicating
                if the reached state is the last of the trajectory;
            gamma (float): the discount factor of the considered problem;
            lam (float): the value for the lamba coefficient used by GEA
                algorithm.
        Returns:
            The new estimate for the value function of the next state
            and the estimated generalized advantage.
        """
        with torch.no_grad():
            v = V(s, pi_h, lengths=lengths, action_history=action_history)
            v_next = V(ss, pi_hn, lengths=lengths, action_history=next_action_history)
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

    def _transform_to_sequences(self, states_old, states, policy_states, actions, next_states, policy_next_states,
                                prev_actions, last, absorbing):
        device = TorchUtils.get_device()

        with torch.no_grad():
            # array preallocation
            s_old = torch.empty(len(states), self._truncation_length, *states_old.shape[1:], device=device)
            s = torch.empty(len(states), self._truncation_length, *states.shape[1:], device=device)
            ps = torch.empty(len(states), *policy_states.shape[1:], device=device)
            a = torch.empty(len(actions), self._truncation_length, *actions.shape[1:], device=device)
            ss = torch.empty(len(states), self._truncation_length, *next_states.shape[1:], device=device)
            pss = torch.empty(len(states), *policy_states.shape[1:], device=device)
            pa = torch.empty(len(states), self._truncation_length, *prev_actions.shape[1:], device=device) \
                if self._history_manager.uses_action else None
            # the sequence lengths stay on the CPU, as pack_padded_sequence requires
            lengths = torch.empty(len(states), dtype=torch.long)

            for i in range(len(states)):
                # determine the beginning of a sequence
                begin_seq = max(i - self._truncation_length + 1, 0)
                end_seq = i + 1

                # the sequence may contain more than one trajectory, we need to cut it so that it contains only one
                lasts_absorbing = last[begin_seq - 1: i].int() + absorbing[begin_seq - 1: i].int()
                begin_traj = torch.where(lasts_absorbing > 0)
                sequence_is_shorter_than_requested = len(*begin_traj) > 0
                if sequence_is_shorter_than_requested:
                    begin_seq = begin_seq + begin_traj[0][-1]

                # get the sequences
                states_old_seq = states_old[begin_seq:end_seq]
                states_seq = states[begin_seq:end_seq]
                actions_seq = actions[begin_seq:end_seq]
                next_states_seq = next_states[begin_seq:end_seq]

                # apply padding
                length_seq = len(states_seq)
                padded_states_old = torch.concatenate([states_old_seq,
                                                       torch.zeros((self._truncation_length - states_old_seq.shape[0],
                                                                    *states_old_seq.shape[1:]), device=device)])
                padded_states = torch.concatenate([states_seq,
                                                   torch.zeros((self._truncation_length - states_seq.shape[0],
                                                                *states_seq.shape[1:]), device=device)])
                padded_next_states = torch.concatenate([next_states_seq,
                                                        torch.zeros((self._truncation_length - next_states_seq.shape[0],
                                                                     *next_states_seq.shape[1:]), device=device)])
                padded_action_seq = torch.concatenate([actions_seq,
                                                       torch.zeros((self._truncation_length - actions_seq.shape[0],
                                                                    *actions_seq.shape[1:]), device=device)])

                s_old[i] = padded_states_old
                s[i] = padded_states
                ps[i] = policy_states[begin_seq]
                a[i] = padded_action_seq
                ss[i] = padded_next_states
                pss[i] = policy_next_states[begin_seq]

                if self._history_manager.uses_action:
                    prev_action_seq = prev_actions[begin_seq:end_seq]
                    pa[i] = torch.concatenate([prev_action_seq,
                                               torch.zeros((self._truncation_length - prev_action_seq.shape[0],
                                                            *prev_action_seq.shape[1:]), device=device)])

                lengths[i] = length_seq

            return s_old, s, ps, a, ss, pss, pa, lengths

    def _update_policy(self, obs, pi_h, act, lengths, adv, old_log_p, prev_action):
        for epoch in range(self._n_epochs_policy()):
            if self._history_manager.uses_action:
                batches = minibatch_generator(self._batch_size(), obs, pi_h, act, lengths, adv, old_log_p, prev_action)
            else:
                batches = minibatch_generator(self._batch_size(), obs, pi_h, act, lengths, adv, old_log_p)
            for batch in batches:
                if self._history_manager.uses_action:
                    obs_i, pi_h_i, act_i, length_i, adv_i, old_log_p_i, prev_action_i = batch
                else:
                    obs_i, pi_h_i, act_i, length_i, adv_i, old_log_p_i = batch
                    prev_action_i = None
                self._optimizer.zero_grad()
                log_p = self.policy.log_prob(obs_i, act_i, pi_h_i, length_i, action_history=prev_action_i)
                prob_ratio = torch.exp(log_p - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy(obs_i)
                loss.backward()
                self._optimizer.step()

    def _post_load(self):
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))
