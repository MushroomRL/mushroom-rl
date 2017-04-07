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
            agent (object): the agent moving according to a policy.
            mdp (object): the environment in which the agent moves.
            params (dict): other params.
        """
        self.agent = agent
        self.mdp = mdp
        self.gamma = params.pop('gamma', mdp.gamma)

        self.logger = logging.getLogger('logger')

        self.fit_params = params.pop('fit_params', dict())

        self.state = self.mdp.reset()
        self._dataset = list()

    def learn(self, n_iterations, how_many, n_fit_steps, iterate_over,
              initial_dataset_size=None, render=False):
        """
        TODO: da migliorare

        This function is used to learn a policy. An iteration of the loop
        consists in collecting a dataset and fitting the agent's Q-function
        approximator on that. Multiple iterations can be done in order to append
        new samples to the dataset using the newly learned policies. This
        function generalizes the learning procedure of online and batch
        algorithms.

        # Arguments
            n_iterations (int > 0): number of iterations.
            how_many (int > 0): number of samples or episodes to collect in a
                single iteration of the loop.
            n_fit_steps (int > 0): number of fitting steps of the learning
                algorithm.
            iterate_over (string): whether to collect samples or episodes in a
                single iteration of the loop.
            render (bool): whether to render the environment or not.
        """
        self.logger.info('*** LEARN ***')
        self.logger.info('Algorithm: ' + str(self))
        self.logger.info('Approximator: ' + str(self.agent.approximator))
        self.logger.info('Environment: ' + str(self.mdp))
        self.logger.info('Policy: ' + str(self.agent.policy))

        assert iterate_over == 'samples' or iterate_over == 'episodes'

        if initial_dataset_size is not None:
            self.move(initial_dataset_size, iterate_over, collect=True,
                      render=render)
            self.fit(n_fit_steps)

        for self.iteration in range(n_iterations):
            self.apply_updates()

            self.move(how_many, iterate_over, collect=True, render=render)
            self.fit(n_fit_steps)

    def evaluate(self, initial_states, render=False):
        """
        This function is used to evaluate the learned policy.

        # Arguments
            initial_states (np.array): the array of initial states from where to
                start the evaluation episodes. An evaluation episode is run for
                each state.
            render (bool): whether to render the environment or not.

        # Returns
            The np.array of discounted rewards obtained in the episodes started
            from the provided initial states.
        """
        self.logger.info('*** EVALUATION ***')
        self.logger.info('Algorithm: ' + str(self))
        self.logger.info('Approximator: ' + str(self.agent.approximator))
        self.logger.info('Environment: ' + str(self.mdp))
        self.logger.info('Policy: ' + str(self.agent.policy))
        self.logger.info('Number of evaluation episodes: ' +
                         str(initial_states.shape[0]))

        Js = list()
        for i in range(initial_states.shape[0]):
            self.state = self.mdp.reset(initial_states[i, :])
            J = self.move(1, 'episodes', force_max_action=True, render=render)
            Js.append(J)

        return np.array(Js).ravel()

    def move(self,
             how_many,
             iterate_over,
             force_max_action=False,
             collect=False,
             render=False):
        """
        Move the agent.

        # Arguments
            how_many (int > 0): number of samples or episodes to collect.
            iterate_over (string): whether to collect samples or episodes.
            force_max_action (bool): whether to perform the greedy action given
                by the policy or not.
            collect (bool): whether to store the collected data or not.
            render (bool): whether to render the environment or not.

        # Returns
            The list of discounted rewards obtained in each episode.
        """
        Js = list()
        i = 0
        n_steps = 0
        n_samples = 0

        if iterate_over == 'episodes':
            self.logger.info('Episodes: %d' % (i + 1))
            self.logger.info(self.state)
        while i < how_many:
            J = 0.
            action = self.agent.draw_action(self.state,
                                            absorbing=False,
                                            force_max_action=force_max_action)
            next_state, reward, absorbing, _ = self.mdp.step(action)
            J += self.gamma ** n_steps * reward
            n_steps += 1

            if render:
                self.mdp.render()

            last = 0 if n_steps < self.mdp.horizon and not absorbing else 1
            sample = self.state.ravel().tolist() + action.ravel().tolist() + \
                     [reward] + next_state.ravel().tolist() + \
                     [absorbing, last]
            n_samples += 1

            self.logger.debug((self.state,
                               action,
                               reward,
                               next_state,
                               absorbing))

            if collect:
                self._dataset.append(sample)

            self.state = next_state

            if last or absorbing:
                if iterate_over == 'episodes':
                    self.logger.info((self.state, reward, absorbing))

                self.state = self.mdp.reset()
                i += 1
                n_steps = 0

                Js.append(J)

                if iterate_over == 'episodes':
                    if i < how_many:
                        self.logger.info('Episode: %d' % (i + 1))
                        self.logger.info(self.state)

            else:
                if iterate_over == 'samples':
                    i += 1

        if iterate_over == 'episodes':
            self.logger.info('Number of samples gathered: ' + str(n_samples))
        else:
            self.logger.debug('Number of samples gathered: ' + str(n_samples))

        return Js

    def apply_updates(self):
        self.agent.policy.update()
        self.updates()

    def get_dataset(self):
        """
        Return the stored dataset.

        # Returns
            The np.array of the stored dataset.
        """
        return np.array(self._dataset)

    def reset_dataset(self):
        """
        Reset the stored dataset list.
        """
        self._dataset = list()
