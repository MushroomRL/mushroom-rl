

def test_imports():
    import mushroom_rl

    import mushroom_rl.algorithms
    import mushroom_rl.algorithms.actor_critic
    import mushroom_rl.algorithms.actor_critic.classic_actor_critic
    import mushroom_rl.algorithms.actor_critic.deep_actor_critic
    import mushroom_rl.algorithms.policy_search
    import mushroom_rl.algorithms.policy_search.black_box_optimization
    import mushroom_rl.algorithms.policy_search.policy_gradient
    import mushroom_rl.algorithms.value
    import mushroom_rl.algorithms.value.batch_td
    import mushroom_rl.algorithms.value.td
    import mushroom_rl.algorithms.value.dqn

    import mushroom_rl.approximators
    import mushroom_rl.approximators._implementations
    import mushroom_rl.approximators.parametric

    import mushroom_rl.core

    import mushroom_rl.distributions

    import mushroom_rl.environments
    import mushroom_rl.environments.generators

    try:
        import mujoco
    except ImportError:
        pass
    else:
        import mushroom_rl.environments.mujoco_envs

    import mushroom_rl.features
    import mushroom_rl.features._implementations
    import mushroom_rl.features.basis
    import mushroom_rl.features.tensors
    import mushroom_rl.features.tiles

    import mushroom_rl.policy

    import mushroom_rl.solvers

    import mushroom_rl.utils

