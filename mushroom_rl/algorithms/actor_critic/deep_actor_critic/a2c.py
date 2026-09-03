import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.value_functions import compute_advantage_montecarlo
from mushroom_rl.rl_utils.parameters import Parameter

from copy import deepcopy


class A2C(DeepAC):
    """
    Advantage Actor Critic algorithm (A2C).
    Synchronous version of the A3C algorithm.
    "Asynchronous Methods for Deep Reinforcement Learning".
    Mnih V. et al. 2016.

    """

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 ent_coeff, max_grad_norm=None, critic_fit_params=None, history_length=1, action_history_length=0):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm;
            actor_optimizer (dict): parameters to specify the actor optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to build;
            ent_coeff ([float, Parameter], 0): coefficient for the entropy penalty;
            max_grad_norm (float, None): maximum norm for gradient clipping. If None, no clipping will be performed,
                unless specified otherwise in actor_optimizer;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator;
            history_length (int, 1): number of consecutive observations stacked as policy input;
            action_history_length (int, 0): number of previous actions fed to the actor and critic.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._entropy_coeff = Parameter.make(ent_coeff, backend='torch')

        self._V = TorchApproximator(**critic_params)

        if 'clipping' not in actor_optimizer and max_grad_norm is not None:
            actor_optimizer = deepcopy(actor_optimizer)
            clipping_params = dict(max_norm=max_grad_norm, norm_type=2)
            actor_optimizer['clipping'] = dict(method=torch.nn.utils.clip_grad_norm_, params=clipping_params)

        super().__init__(mdp_info, policy, actor_optimizer, policy.parameters(), history_length=history_length,
                         action_history_length=action_history_length)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _entropy_coeff='mushroom',
            _V='mushroom'
        )
        self._add_logger_attr('_V', group='critic')

    def fit(self, dataset):
        self._history_manager.update_preprocessors(dataset)

        state, action, reward, next_state, absorbing, last, extra = self._history_manager.parse_history(dataset)
        prev_action = extra.get('action_history')

        v, adv = compute_advantage_montecarlo(self._V, state, next_state, reward, absorbing, last, self.mdp_info.gamma,
                                              action_history=prev_action, action=action)
        self._V.fit(state, v, action_history=prev_action, **self._critic_fit_params)

        loss = self._loss(state, action, adv, prev_action)
        self._optimize_actor_parameters(loss)

        if self._logger:
            self._logger.log_training('actor', loss=loss.item(), entropy=self.policy.entropy(state).item())
            self._logger.advance_step()

    def _loss(self, state, action, adv, action_history=None):
        gradient_loss = -torch.mean(self.policy.log_prob(state, action, action_history=action_history) * adv)
        entropy_loss = -self.policy.entropy(state)

        return gradient_loss + self._entropy_coeff() * entropy_loss

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
