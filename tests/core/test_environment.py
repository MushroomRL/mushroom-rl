from mushroom_rl.core import Environment, register_env


def test_register_env_registers_and_returns_the_class():
    class RegisterEnvTestEnv(Environment):
        pass

    try:
        decorated = register_env(RegisterEnvTestEnv)

        assert decorated is RegisterEnvTestEnv
        assert 'RegisterEnvTestEnv' in Environment.list_registered()
        assert Environment._registered_envs['RegisterEnvTestEnv'] is RegisterEnvTestEnv
    finally:
        Environment._registered_envs.pop('RegisterEnvTestEnv', None)


def test_register_env_does_not_replace_a_registered_name():
    def make_env_class():
        class RegisterEnvDuplicateEnv(Environment):
            pass

        return RegisterEnvDuplicateEnv

    first = make_env_class()
    second = make_env_class()

    try:
        register_env(first)
        decorated = register_env(second)

        assert decorated is second
        assert Environment._registered_envs['RegisterEnvDuplicateEnv'] is first
    finally:
        Environment._registered_envs.pop('RegisterEnvDuplicateEnv', None)


def test_base_environment_classes_are_not_registered():
    base_classes = ['Environment', 'VectorizedEnvironment', 'MultiprocessEnvironment', 'MuJoCo', 'MultiMuJoCo',
                    'MuJoCoWarp', 'PyBullet', 'IsaacSim', 'QuadrupedIsaac', 'MiniGridBase', 'AirHockeyBase',
                    'AirHockeySingle', 'AirHockeyDouble', 'Panda', 'AirHockeyBaseBullet', 'AirHockeySingleBullet']

    registered = Environment.list_registered()

    for name in base_classes:
        assert name not in registered


def test_environments_without_extras_are_registered():
    environments = ['CarOnHill', 'CartPole', 'FiniteMDP', 'GridWorld', 'GridWorldVanHasselt', 'InvertedPendulum', 'LQR',
                    'PuddleWorld', 'Segway', 'ShipSteering', 'SimpleChain', 'Taxi']

    registered = Environment.list_registered()

    for name in environments:
        assert name in registered
