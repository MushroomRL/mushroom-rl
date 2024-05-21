import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import Box, Discrete

from mushroom_rl.utils.viewer import Viewer


class RoomToyEnv(Environment):
    def __init__(self, size=5., goal=(2.5, 2.5), goal_radius=0.6):

        # Save important environment information
        self._size = size
        self._goal = np.array(goal)
        self._goal_radius = goal_radius

        # Create the action space.
        action_space = Discrete(4)  # 4 actions: N, S, W, E

        # Create the observation space. It's a 2D box of dimension (size x size).
        # You can also specify low and high array, if every component has different limits
        shape = (2,)
        observation_space = Box(0, size, shape)

        # Create the MDPInfo structure, needed by the environment interface
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=100, dt=0.1)

        super().__init__(mdp_info)

        # Create a state class variable to store the current state
        self._state = None

        # Create the viewer
        self._viewer = Viewer(size, size)

    def reset(self, state=None):

        if state is None:
            # Generate randomly a state inside the state space, but not inside the goal
            self._state = np.random.rand(2) * self._size

            # Check if it's inside the goal radius and repeat the sample if necessary
            while np.linalg.norm(self._state - self._goal) < self._goal_radius:
                self._state = np.random.rand(2) * self._size
        else:
            # If an initial state is provided, set it and return, after checking it's valid.
            assert np.all(state < self._size) and np.all(state > 0)
            assert np.linalg.norm(state - self._goal) > self._goal_radius
            self._state = state

        # Return the current state
        return self._state

    def step(self, action):
        # convert the action in a N, S, W, E movement
        movement = np.zeros(2)
        if action == 0:
            movement[1] += 0.1
        elif action == 1:
            movement[1] -= 0.1
        elif action == 2:
            movement[0] -= 0.1
        elif action == 3:
            movement[0] += 0.1
        else:
            raise ValueError('The environment has only 4 actions')

        # Apply the movement with some noise:
        self._state += movement + np.random.randn(2)*0.05

        # Clip the state space inside the boundaries.
        low = self.info.observation_space.low
        high = self.info.observation_space.high

        self._state = Environment._bound(self._state, low, high)

        # Compute distance form goal
        goal_distance = np.linalg.norm(self._state - self._goal)

        # Compute the reward as distance penalty from goal
        reward = -goal_distance

        # Set the absorbing flag if goal is reached
        absorbing = goal_distance < self._goal_radius

        # Return all the information + empty dictionary (used to pass additional information)
        return self._state, reward, absorbing, {}

    def render(self, record=False):
        # Draw a red circle for the agent
        self._viewer.circle(self._state, 0.1, color=(255, 0, 0))

        # Draw a green circle for the goal
        self._viewer.circle(self._goal, self._goal_radius, color=(0, 255, 0))

        # Get the image if the record flag is set to true
        frame = self._viewer.get_frame() if record else None

        # Display the image for the control time (0.1 seconds)
        self._viewer.display(self.info.dt)

        return frame


# Register the class
RoomToyEnv.register()

if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.features import Features
    from mushroom_rl.features.tiles import Tiles
    from mushroom_rl.rl_utils.parameters import Parameter

    # Set the seed
    np.random.seed(1)

    # Create the toy environment with default parameters
    env = Environment.make('RoomToyEnv')

    # Using an epsilon-greedy policy
    epsilon = Parameter(value=0.1)
    pi = EpsGreedy(epsilon=epsilon)

    # Creating a simple agent using linear approximator with tiles
    n_tilings = 5
    tilings = Tiles.generate(n_tilings, [10, 10],
                             env.info.observation_space.low,
                             env.info.observation_space.high)
    features = Features(tilings=tilings)

    learning_rate = Parameter(.1 / n_tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(env.info.action_space.n,),
                               n_actions=env.info.action_space.n)

    agent = TrueOnlineSARSALambda(env.info, pi,
                                  approximator_params=approximator_params,
                                  features=features,
                                  learning_rate=learning_rate,
                                  lambda_coeff=.9)

    # Reinforcement learning experiment
    core = Core(agent, env)

    # Visualize initial policy for 3 episodes
    dataset = core.evaluate(n_episodes=3, render=True)

    # Print the average objective value before learning
    J = np.mean(dataset.discounted_return)
    print(f'Objective function before learning: {J}')

    # Train
    core.learn(n_steps=20000, n_steps_per_fit=1, render=False)

    # Visualize results for 3 episodes
    dataset = core.evaluate(n_episodes=3, render=True)

    # Print the average objective value after learning
    J = np.mean(dataset.discounted_return)
    print(f'Objective function after learning: {J}')



