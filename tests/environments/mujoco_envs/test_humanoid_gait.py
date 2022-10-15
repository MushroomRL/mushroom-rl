try:
    raise ImportError("Skip Test")
    import numpy as np

    from mushroom_rl.environments.mujoco_envs import HumanoidGait
    from mushroom_rl.environments.mujoco_envs.humanoid_gait import \
        VelocityProfile3D, RandomConstantVelocityProfile, ConstantVelocityProfile


    def create_mdp(goal, gamma, horizon):
        if goal == "trajectory":
            mdp = HumanoidGait(gamma=gamma, horizon=horizon, n_intermediate_steps=10,
                               goal_reward="trajectory",
                               goal_reward_params=dict(use_error_terminate=True),
                               use_muscles=True,
                               obs_avg_window=1, act_avg_window=1)

        elif goal == "max_vel":
            mdp = HumanoidGait(gamma=gamma, horizon=horizon, n_intermediate_steps=10,
                               goal_reward="max_vel",
                               goal_reward_params=dict(traj_start=True),
                               use_muscles=False,
                               obs_avg_window=1, act_avg_window=1)

        elif goal == "vel_profile":
            velocity_profile = dict(profile_instance=VelocityProfile3D([
                    RandomConstantVelocityProfile(min=0.5, max=2.0),
                    ConstantVelocityProfile(0),
                    ConstantVelocityProfile(0)]))

            mdp = HumanoidGait(gamma=gamma, horizon=horizon, n_intermediate_steps=10,
                               goal_reward="vel_profile",
                               goal_reward_params=dict(traj_start=True,
                                                       **velocity_profile),
                               use_muscles=False,
                               obs_avg_window=1, act_avg_window=1)
        return mdp


    def test_humanoid_gait():
        np.random.seed(1)

        # MDP
        gamma = 0.99
        horizon = 2000
        for goal in ["trajectory", "vel_profile", "max_vel"]:
            mdp = create_mdp(goal, gamma, horizon)

            s = 0
            done = True
            mdp.reset()
            while s < 100:
                if done:
                    mdp.reset()

                random_action = np.random.uniform(low=mdp.info.action_space.low / 2.0,
                                                  high=mdp.info.action_space.high / 2.0,
                                                  size=mdp.info.action_space.shape[0])
                obs, reward, done, _ = mdp.step(random_action)
                s += 1
except ImportError:
    pass

