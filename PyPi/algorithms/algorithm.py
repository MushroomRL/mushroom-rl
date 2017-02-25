class Algorithm(object):
    def __init__(self, agent, mdp, logger, **params):
        self.agent = agent
        self.mdp = mdp
        self.logger = logger
        self.gamma = params.pop('gamma')

    def run(self, n_episodes):
        self.logger.info(self.__name__ + '; ' + self.mdp.__name__)
        self.logger.info('horizon = ' + str(n_episodes) +
                         '; gamma = ' + str(self.gamma))

        for i in range(n_episodes):
            self.logger.info('Episode: ' + str(i))

            state = self.mdp.reset()
            absorbing = False

            i = 0
            while not absorbing and i < self.mdp.horizon:
                action = self.agent.draw_action(state, absorbing)
                next_state, reward, absorbing, _ = self.mdp.step(action)

                self.logger.info((state, action, next_state, reward))

                self.step(state, action, reward, next_state, absorbing)

                state = next_state
                i += 1
