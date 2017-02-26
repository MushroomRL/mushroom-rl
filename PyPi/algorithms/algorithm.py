import numpy as np


class Algorithm(object):
    """
    Implements the functions to run an algorithm.
    """
    def __init__(self, agent, mdp, logger, **params):
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
        self.logger = logger
        self.gamma = params.pop('gamma')

    def run(self, n_episodes, evaluate=False):
        """
        Runs the algorithm.

        # Arguments
            n_episodes (int > 0): number of episodes to run
            evaluate (bool): whether to perform a test episode or not
        """
        phase = 'Train' if not evaluate else 'Test'
        self.logger.info('*** ' + phase + ' ***')
        self.logger.info(self.__name__ + '; ' + self.mdp.__name__)
        self.logger.info('horizon = ' + str(n_episodes) +
                         '; gamma = ' + str(self.gamma))

        Js = list()
        for i in range(n_episodes):
            self.logger.info('Episode: ' + str(i))

            state = self.mdp.reset()
            absorbing = False

            self.logger.info('Initial state: ' + str(state))

            J = 0.
            i = 0
            while not absorbing and i < self.mdp.horizon:
                action = self.agent.draw_action(state, absorbing)
                next_state, reward, absorbing, _ = self.mdp.step(action)

                self.logger.debug((state, action, next_state, reward))

                if not evaluate:
                    self.step(state, action, reward, next_state, absorbing)

                state = next_state

                J += self.gamma**i * reward
                i += 1

            Js.append(J)

            self.logger.info(('End state: ' + str(state) +
                              '; ' + str(absorbing) +
                              '; J: ' + str(J)))

        return np.array(Js)
