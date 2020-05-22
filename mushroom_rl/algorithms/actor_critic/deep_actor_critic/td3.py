import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DDPG
from mushroom_rl.policy import Policy


class TD3(DDPG):
    """
    Twin Delayed DDPG algorithm.
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    """
    def __init__(self, mdp_info, policy_class, policy_params, actor_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay=2,
                 noise_std=.2, noise_clip=.5, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau (float): value of coefficient for soft updates;
            policy_delay (int, 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std (float, .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip (float, .5): maximum absolute value for policy smoothing
                noise;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._noise_std = noise_std
        self._noise_clip = noise_clip

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2

        self._add_save_attr(
            _noise_std='primitive',
            _noise_clip='primitive'
        )

        super().__init__(mdp_info, policy_class, policy_params,  actor_params,
                         actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, tau,
                         policy_delay, critic_fit_params)

    def _loss(self, state):
        action = self._actor_approximator(state, output_tensor=True)
        q = self._critic_approximator(state, action, idx=0, output_tensor=True)

        return -q.mean()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self._target_actor_approximator(next_state)

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        eps = np.random.normal(scale=self._noise_std, size=a.shape)
        eps_clipped = np.clip(eps, -self._noise_clip, self._noise_clip)
        a_smoothed = np.clip(a + eps_clipped, low, high)

        q = self._target_critic_approximator.predict(next_state, a_smoothed,
                                                     prediction='min')
        q *= 1 - absorbing

        return q
