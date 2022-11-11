import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.value_functions import compute_advantage_montecarlo
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.torch import to_float_tensor

from copy import deepcopy


class A2C(DeepAC):
    """
    Advantage Actor Critic algorithm (A2C).
    Synchronous version of the A3C algorithm.
    "Asynchronous Methods for Deep Reinforcement Learning".
    Mnih V. et. al.. 2016.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 ent_coeff, max_grad_norm=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            ent_coeff ([float, Parameter], 0): coefficient for the entropy penalty;
            max_grad_norm (float, None): maximum norm for gradient clipping.
                If None, no clipping will be performed, unless specified
                otherwise in actor_optimizer;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._entropy_coeff = to_parameter(ent_coeff)

        self._V = Regressor(TorchApproximator, **critic_params)

        if 'clipping' not in actor_optimizer and max_grad_norm is not None:
            actor_optimizer = deepcopy(actor_optimizer)
            clipping_params = dict(max_norm=max_grad_norm, norm_type=2)
            actor_optimizer['clipping'] = dict(
                method=torch.nn.utils.clip_grad_norm_, params=clipping_params)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _entropy_coeff='mushroom',
            _V='mushroom'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy.parameters())

    def fit(self, dataset, **info):
        state, action, reward, next_state, absorbing, _ = parse_dataset(dataset)

        v, adv = compute_advantage_montecarlo(self._V, state, next_state,
                                              reward, absorbing,
                                              self.mdp_info.gamma)
        self._V.fit(state, v, **self._critic_fit_params)

        loss = self._loss(state, action, adv)
        self._optimize_actor_parameters(loss)

    def _loss(self, state, action, adv):
        use_cuda = self.policy.use_cuda

        s = to_float_tensor(state, use_cuda)
        a = to_float_tensor(action, use_cuda)

        adv_t = to_float_tensor(adv, use_cuda)

        gradient_loss = -torch.mean(self.policy.log_prob_t(s, a)*adv_t)
        entropy_loss = -self.policy.entropy_t(s)

        return gradient_loss + self._entropy_coeff() * entropy_loss

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
