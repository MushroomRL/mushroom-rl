from mushroom.algorithms.agent import Agent

import numpy as np


class PolicyGradient(Agent):
    """
    Abstract class to implement a generic Policy Search algorithm using the
    gradient of the policy to update its parameters.
    "A survey on Policy Gradient algorithms for Robotics". Deisenroth M. P. et
    al.. 2011.

    """
    def __init__(self, policy, mdp_info, learning_rate, features):
        """
        Constructor.

        Args:
            learning_rate (float): the learning rate.

        """
        self.learning_rate = learning_rate
        self.df = 1
        self.J_episode = 0

        super().__init__(policy, mdp_info, features)

    def fit(self, dataset):
        J = list()
        self.df = 1.
        self.J_episode = 0.
        self._init_update()
        for sample in dataset:
            x, u, r, xn, _, last = self._parse(sample)
            self._step_update(x, u, r)
            self.J_episode += self.df * r
            self.df *= self.mdp_info.gamma

            if last:
                self._episode_end_update()
                J.append(self.J_episode)
                self.J_episode = 0.
                self.df = 1.
                self._init_update()

        self._update_parameters(J)

    def _update_parameters(self, J):
        """
        Update the parameters of the policy.

        Args:
             J (list): list of the cumulative discounted rewards for each
                episode in the dataset.

        """
        res = self._compute_gradient(J)

        theta = self.policy.get_weights()

        if len(res) == 1:
            grad = res[0]
            delta = self.learning_rate(grad) * grad
        else:
            grad, nat_grad = res
            delta = self.learning_rate(grad, nat_grad) * nat_grad

        theta_new = theta + delta
        self.policy.set_weights(theta_new)

    def _init_update(self):
        """
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE resets some data structure).

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _step_update(self, x, u, r):
        """
        This function is called, when parsing the dataset, at each episode step.

        Args:
            x (np.ndarray): the state at the current step;
            u (np.ndarray): the action at the current step;
            r (np.ndarray): the reward at the current step.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _episode_end_update(self, J_episode):
        """
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE updates some data structures).

        Args:
            J_episode (float): cumulative discounted reward of the current
                episode.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _compute_gradient(self, J):
        """
        Return the gradient computed by the algorithm.

        Args:
             J (list): list of the cumulative discounted rewards for each
                episode in the dataset.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _parse(self, sample):
        """
        Utility to parse the sample.

        Args:
             sample (list): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag. If provided, ``state`` is preprocessed with the features.

        """
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        last = sample[5]

        if self.phi is not None:
            state = self.phi(state)

        return state, action, reward, next_state, absorbing, last


class REINFORCE(PolicyGradient):
    """
    REINFORCE algorithm.
    "Simple Statistical Gradient-Following Algorithms for Connectionist
    Reinforcement Learning", Williams R. J.. 1992.

    """
    def __init__(self, policy, mdp_info, learning_rate, features=None):
        super().__init__(policy, mdp_info, learning_rate, features)
        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.baseline_num = list()
        self.baseline_den = list()

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

    def _compute_gradient(self, J):
        baseline = np.mean(self.baseline_num, axis=0) / np.mean(
            self.baseline_den, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.
        grad_J_episode = list()
        for i, J_episode in enumerate(J):
            sum_d_log_pi = self.list_sum_d_log_pi[i]
            grad_J_episode.append(sum_d_log_pi * (J_episode - baseline))

        grad_J = np.mean(grad_J_episode, axis=0)
        self.list_sum_d_log_pi = list()
        self.baseline_den = list()
        self.baseline_num = list()

        return grad_J,

    def _step_update(self, x, u, r):
        d_log_pi = self.policy.diff_log(x, u)
        self.sum_d_log_pi += d_log_pi

    def _episode_end_update(self):
        self.list_sum_d_log_pi.append(self.sum_d_log_pi)
        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)
        self.baseline_num.append(squared_sum_d_log_pi * self.J_episode)
        self.baseline_den.append(squared_sum_d_log_pi)

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)


class GPOMDP(PolicyGradient):
    """
    GPOMDP algorithm.
    "Infinite-Horizon Policy-Gradient Estimation". Baxter J. and Bartlett P. L..
    2001.

    """
    def __init__(self, policy, mdp_info, learning_rate, features=None):
        super().__init__(policy, mdp_info, learning_rate, features)

        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.list_sum_d_log_pi_ep = list()

        self.list_reward = list()
        self.list_reward_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        self.step_count = 0

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

    def _compute_gradient(self, J):
        gradient = np.zeros(self.policy.weights_size)

        n_episodes = len(self.list_sum_d_log_pi_ep)

        for i in range(n_episodes):
            list_sum_d_log_pi = self.list_sum_d_log_pi_ep[i]
            list_reward = self.list_reward_ep[i]

            n_steps = len(list_sum_d_log_pi)

            for t in range(n_steps):
                step_grad = list_sum_d_log_pi[t]
                step_reward = list_reward[t]
                baseline = self.baseline_num[t] / self.baseline_den[t]
                baseline[np.logical_not(np.isfinite(baseline))] = 0.
                gradient += (step_reward - baseline) * step_grad

        gradient /= n_episodes

        self.list_reward_ep = list()
        self.list_sum_d_log_pi_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        return gradient,

    def _step_update(self, x, u, r):
        discounted_reward = self.df*r
        self.list_reward.append(discounted_reward)

        d_log_pi = self.policy.diff_log(x, u)
        self.sum_d_log_pi += d_log_pi

        self.list_sum_d_log_pi.append(self.sum_d_log_pi)

        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)

        if self.step_count < len(self.baseline_num):
            self.baseline_num[
                self.step_count] += discounted_reward * squared_sum_d_log_pi
            self.baseline_den[self.step_count] += squared_sum_d_log_pi
        else:
            self.baseline_num.append(discounted_reward * squared_sum_d_log_pi)
            self.baseline_den.append(squared_sum_d_log_pi)

        self.step_count += 1

    def _episode_end_update(self):
        self.list_reward_ep.append(self.list_reward)
        self.list_reward = list()

        self.list_sum_d_log_pi_ep.append(self.list_sum_d_log_pi)
        self.list_sum_d_log_pi = list()

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)
        self.list_sum_d_log_pi = list()
        self.step_count = 0


class eNAC(PolicyGradient):
    """
    ENAC algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J. 2013.

    """
    def __init__(self, policy, mdp_info, learning_rate, features=None,
                 critic_features=None):
        """
        Constructor.

        Args:
            critic_features (Features, None): features used by the critic.

        """
        super().__init__(policy, mdp_info, learning_rate, features)
        self.phi_c = critic_features

        self.sum_grad_log = None
        self.psi_ext = None
        self.sum_grad_log_list = list()

    def _compute_gradient(self, J):
        R = np.array(J)
        PSI = np.array(self.sum_grad_log_list)

        w_and_v = np.linalg.pinv(PSI).dot(R)
        nat_grad = w_and_v[:self.policy.weights_size]

        self.sum_grad_log_list = list()

        return nat_grad,

    def _step_update(self, x, u, r):
        self.sum_grad_log += self.policy.diff_log(x, u)

        if self.psi_ext is None:
            if self.phi_c is None:
                self.psi_ext = np.ones(1)
            else:
                self.psi_ext = self.phi_c(x)

    def _episode_end_update(self):
        psi = np.concatenate((self.sum_grad_log, self.psi_ext))
        self.sum_grad_log_list.append(psi)

    def _init_update(self):
        self.psi_ext = None
        self.sum_grad_log = np.zeros(self.policy.weights_size)
