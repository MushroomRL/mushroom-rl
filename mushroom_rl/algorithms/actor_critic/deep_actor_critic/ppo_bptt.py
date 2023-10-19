import torch

from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import update_optimizer_parameters
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor


class PPO_BPTT(Agent):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, dim_env_state, ent_coeff=0.0,
                 critic_fit_params=None, truncation_length=5):
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

        self._truncation_length = truncation_length
        self._dim_env_state = dim_env_state

        self._iter = 1

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom',
            _iter='primitive',
            _dim_env_state='primitive'
        )

        super().__init__(mdp_info, policy, None)

        # add the standardization preprocessor
        self._preprocessors.append(StandardizationPreprocessor(mdp_info))

    def divide_state_to_env_hidden_batch(self, states):
        assert len(states.shape) > 1, "This function only divides batches of states."
        return states[:, 0:self._dim_env_state], states[:, self._dim_env_state:]

    def fit(self, dataset, **info):
        obs, act, r, obs_next, absorbing, last = dataset.parse(to='torch')
        policy_state, policy_next_state = dataset.parse_policy_state(to='torch')
        obs_seq, policy_state_seq, act_seq, obs_next_seq, policy_next_state_seq, lengths = \
            self.transform_to_sequences(obs, policy_state, act, obs_next, policy_next_state, last, absorbing)

        v_target, adv = self.compute_gae(self._V, obs_seq, policy_state_seq, obs_next_seq, policy_next_state_seq,
                                         lengths, r, absorbing, last, self.mdp_info.gamma, self._lambda())
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        old_pol_dist = self.policy.distribution_t(obs_seq, policy_state_seq, lengths)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(obs_seq, policy_state_seq, lengths, v_target, **self._critic_fit_params)

        self._update_policy(obs_seq, policy_state_seq, act, lengths, adv, old_log_p)

        # Print fit information
        self._log_info(dataset, obs_seq, policy_state_seq, lengths, v_target, old_pol_dist)
        self._iter += 1

    def transform_to_sequences(self, states, policy_states, actions, next_states, policy_next_states, last, absorbing):

        s = torch.empty(len(states), self._truncation_length, states.shape[-1])
        ps = torch.empty(len(states), policy_states.shape[-1])
        a = torch.empty(len(actions), self._truncation_length, actions.shape[-1])
        ss = torch.empty(len(states), self._truncation_length, states.shape[-1])
        pss = torch.empty(len(states), policy_states.shape[-1])
        lengths = torch.empty(len(states), dtype=torch.long)

        for i in range(len(states)):
            # determine the begin of a sequence
            begin_seq = max(i - self._truncation_length + 1, 0)
            end_seq = i + 1

            # maybe the sequence contains more than one trajectory, so we need to cut it so that it contains only one
            lasts_absorbing = last[begin_seq - 1: i].int() + absorbing[begin_seq - 1: i].int()
            begin_traj = torch.where(lasts_absorbing > 0)
            sequence_is_shorter_than_requested = len(*begin_traj) > 0
            if sequence_is_shorter_than_requested:
                begin_seq = begin_seq + begin_traj[0][-1]

            # get the sequences
            states_seq = states[begin_seq:end_seq]
            actions_seq = actions[begin_seq:end_seq]
            next_states_seq = next_states[begin_seq:end_seq]

            # apply padding
            length_seq = len(states_seq)
            padded_states = torch.concatenate([states_seq,
                                               torch.zeros((self._truncation_length - states_seq.shape[0],
                                                                  states_seq.shape[1]))])
            padded_next_states = torch.concatenate([next_states_seq,
                                                    torch.zeros((self._truncation_length - next_states_seq.shape[0],
                                                    next_states_seq.shape[1]))])
            padded_action_seq = torch.concatenate([actions_seq,
                                                   torch.zeros((self._truncation_length - actions_seq.shape[0],
                                                                actions_seq.shape[1]))])

            s[i] = padded_states
            ps[i] = policy_states[begin_seq]
            a[i] = padded_action_seq
            ss[i] = padded_next_states
            pss[i] = policy_next_states[begin_seq]

            lengths[i] = length_seq

        return s.detach(), ps.detach(), a.detach(), ss.detach(), pss.detach(), lengths.detach()

    def _update_policy(self, obs, pi_h, act, lengths, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, pi_h_i, act_i, length_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, pi_h, act, lengths, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i, act_i, pi_h_i, length_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy_t(obs_i)
                loss.backward()
                self._optimizer.step()

    def _log_info(self, dataset, x, pi_h, lengths, v_target, old_pol_dist):
        pass

    def _post_load(self):
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))

    @staticmethod
    def compute_gae(V, s, pi_h, ss, pi_hn, lengths, r, absorbing, last, gamma, lam):
        """
        Function to compute Generalized Advantage Estimation (GAE)
        and new value function target over a dataset.

        "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation".
        Schulman J. et al.. 2016.

        Args:
            V (Regressor): the current value function regressor;
            s (numpy.ndarray): the set of states in which we want
                to evaluate the advantage;
            ss (numpy.ndarray): the set of next states in which we want
                to evaluate the advantage;
            r (numpy.ndarray): the reward obtained in each transition
                from state s to state ss;
            absorbing (numpy.ndarray): an array of boolean flags indicating
                if the reached state is absorbing;
            last (numpy.ndarray): an array of boolean flags indicating
                if the reached state is the last of the trajectory;
            gamma (float): the discount factor of the considered problem;
            lam (float): the value for the lamba coefficient used by GEA
                algorithm.
        Returns:
            The new estimate for the value function of the next state
            and the estimated generalized advantage.
        """
        with torch.no_grad():
            v = V(s, pi_h, lengths, output_tensor=True)
            v_next = V(ss, pi_hn, lengths, output_tensor=True)
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
