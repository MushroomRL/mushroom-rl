import logging
import numpy as np


class Algorithm(object):
    """
    Implements the functions to run an algorithm.
    """
    def __init__(self, agent, mdp, **params):
        """
        Constructor.

        # Arguments
            agent (object): the agent moving according to the policy
            mdp (object): the environment in which the agent moves
            logger (object): logger
            params (dict): other params
        """
        self.agent = agent
        self.mdp = mdp
        self.logger = logging.getLogger('logger')
        self.gamma = params.pop('gamma', mdp.gamma)

        self.state = self.mdp.reset()
        self._dataset = list()

    def learn(self, n_iterations, how_many, n_fit_steps, iterate_over):
        """
        TODO: da migliorare

        Runs the algorithm.

        # Arguments
            n_episodes (int > 0): number of episodes to run
            evaluate (bool): whether to perform a test episode or not
        """
        assert iterate_over == 'samples' or iterate_over == 'episodes'
        for i in range(n_iterations):
            self.move(how_many, iterate_over)
            self.fit(n_fit_steps)

    def evaluate(self, initial_states):
        Js = list()
        for i in range(initial_states.shape[0]):
            self.state = self.mdp.reset(initial_states[i, :])
            J = self.move(1, 'episodes', force_max_action=True)
            Js.append(J)

        return Js

    def move(self, how_many, iterate_over, force_max_action=False):
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
        return np.array(self._dataset)

    def reset_dataset(self):
        self._dataset = list()
