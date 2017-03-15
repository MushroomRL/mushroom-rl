import logging
import numpy as np


class Algorithm(object):
    """
    Implements the functions to run a generic algorithm.
    """
    def __init__(self, agent, mdp, **params):
        """
        Constructor.

        # Arguments
            agent (object): the agent moving according to a policy
            mdp (object): the environment in which the agent moves
            params (dict): other params
        """
        self.agent = agent
        self.mdp = mdp
        self.gamma = params.pop('gamma', mdp.gamma)

        self.logger = logging.getLogger('logger')

        self.fit_params = params.pop('fit_params', dict())

        self.state = self.mdp.reset()
        self._dataset = list()

    def learn(self, n_iterations, how_many, n_fit_steps, iterate_over):
        """
        TODO: da migliorare

        This function is used to learn a policy. An iteration of the loop
        consists in collecting a dataset and fitting the agent's Q-approximator
        on that. This function generalizes the learning procedure of online
        and batch algorithms.

        # Arguments
            n_iterations (int > 0): number of iterations
            how_many (int > 0): number of samples or episodes to collect in a
                                single iteration of the loop
            n_fit_steps (int > 0): number of fitting steps of the learning
                                   algorithm
            iterate_over (string): whether to collect samples or episodes in a
                                   single iteration of the loop
        """
        assert iterate_over == 'samples' or iterate_over == 'episodes'
        for i in range(n_iterations):
            self.move(how_many, iterate_over, collect=True)
            self.fit(n_fit_steps)

    def evaluate(self, initial_states):
        """
        This function is used to evaluate the learned policy.

        # Arguments
            initial_states (np.array): the array of initial states from where to
                                       start the evaluation episodes. An
                                       evaluation episode is run for each state

        # Returns
            the list of discounted rewards obtained in the episodes started
            from the provided initial states
        """
        Js = list()
        for i in range(initial_states.shape[0]):
            self.state = self.mdp.reset(initial_states[i, :])
            J = self.move(1, 'episodes', force_max_action=True)
            Js.append(J)

        return np.array(Js).ravel()

    def move(self,
             how_many,
             iterate_over,
             force_max_action=False,
             collect=False):
        """
        Move the agent.

        # Arguments
            how_many (int > 0): number of samples or episodes to collect
            iterate_over (string): whether to collect samples or episodes
            force_max_action (bool): whether to perform the greedy action given
                                     by the policy or not
            collect (bool): whether to store the collected data or not

        # Returns
            the list of discounted rewards obtained in each episode
        """
        Js = list()
        i = 0
        n_steps = 0
        while i < how_many:
            J = 0.
            action = self.agent.draw_action(self.state,
                                            absorbing=False,
                                            force_max_action=force_max_action)
            next_state, reward, absorbing, _ = self.mdp.step(action)

            last = 0 if n_steps < self.mdp.horizon or not absorbing else 1
            sample = self.state.ravel().tolist() + action.ravel().tolist() + \
                     [reward] + next_state.ravel().tolist() + \
                     [absorbing, last]

            self.logger.debug((self.state,
                               action,
                               reward,
                               next_state,
                               absorbing))

            if collect:
                self._dataset.append(sample)

            self.state = next_state

            J += self.gamma**n_steps * reward

            if last or absorbing:
                self.state = self.mdp.reset()
                i += 1
                n_steps = 0

                Js.append(J)
            else:
                n_steps += 1
                if iterate_over == 'samples':
                    i += 1

        return Js

    def get_dataset(self):
        """
        Return the stored dataset.

        # Returns
            the numpy array of the stored dataset
        """
        return np.array(self._dataset)

    def reset_dataset(self):
        """
        Reset the stored dataset list.
        """
        self._dataset = list()
