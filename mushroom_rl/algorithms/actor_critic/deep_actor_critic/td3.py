from functools import cached_property

import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DDPG
from mushroom_rl.rl_utils.parameters import Parameter


class TD3(DDPG):
    """
    Twin Delayed DDPG algorithm.
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al. 2018.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, batch_size, initial_replay_size,
                 max_replay_size, tau, policy_delay=2, noise_std=.2, noise_clip=.5, critic_fit_params=None,
                 history_length=1, device=None):
        """
        Constructor.

        Args:
            policy (HasLocation): policy to be learned by the algorithm;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std ([float, Parameter], .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip ([float, Parameter], .5): maximum absolute value for policy smoothing
                noise;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
            history_length (int, 1): number of consecutive observations stacked as policy input;
            device (str, None): device the agent data is stored on, or ``None`` for the default one.

        """
        self._noise_std = Parameter.make(noise_std, backend='torch', device=device)
        self._noise_clip = Parameter.make(noise_clip, backend='torch', device=device)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] >= 2
        else:
            critic_params['n_models'] = 2

        super().__init__(mdp_info, policy, actor_optimizer, critic_params, batch_size, initial_replay_size,
                         max_replay_size, tau, policy_delay, critic_fit_params, history_length=history_length,
                         device=device)

        self._add_save_attr(
            _noise_std='mushroom',
            _noise_clip='mushroom'
        )

    def _loss(self, batch):
        action = self.policy.location(batch.state)
        q = self._critic_approximator(batch.critic_state, action, idx=0, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, batch):
        """
        Target value of the next states of a batch.

        Args:
            batch (Batch): the fit batch, whose next states are evaluated.

        Returns:
            Action-values returned by the target critic for the next states and the actions returned by the actor,
            zero on the absorbing states.

        """
        a = self._target_policy.location(batch.next_state)

        eps = torch.randn_like(a) * self._noise_std()
        eps_clipped = torch.clamp(eps, -self._noise_clip(), self._noise_clip.get_value())

        a_smoothed = torch.clamp(a + eps_clipped, self._action_low, self._action_high)

        q = self._target_critic_approximator.predict(batch.critic_next_state, a_smoothed,
                                                     prediction='min', **self._critic_predict_params)
        q *= ~batch.absorbing

        return q

    @cached_property
    def _action_low(self):
        return torch.as_tensor(self.mdp_info.action_space.low, device=self._info.device)

    @cached_property
    def _action_high(self):
        return torch.as_tensor(self.mdp_info.action_space.high, device=self._info.device)
