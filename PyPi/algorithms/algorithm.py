class Algorithm(object):
    def __init__(self, agent, mdp, **params):
        self.agent = agent
        self.mdp = mdp
        self.gamma = params.pop('gamma')

    def run(self, n_episodes):
        for i in range(n_episodes):
            state = self.mdp.reset()
            absorbing = False
            i = 0
            while not absorbing and i < self.mdp.horizon:
                action = self.agent.draw_action(state, absorbing)
                next_state, reward, absorbing, _ = self.mdp.step(action)

                self.step(state, action, reward, next_state, absorbing)

                state = next_state
                i += 1
