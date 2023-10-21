from mushroom_rl.policy import Policy, ParametricPolicy


def abstract_method_tester(f, ex, *args):
    try:
        f(*args)
    except ex:
        pass
    else:
        assert False


def test_policy_interface():
    tmp = Policy()
    abstract_method_tester(tmp.__call__, NotImplementedError, None, None, None)
    abstract_method_tester(tmp.draw_action, NotImplementedError, None, None)
    tmp.reset()


def test_parametric_policy():
    tmp = ParametricPolicy()
    abstract_method_tester(tmp.diff_log, RuntimeError, None, None, None)
    abstract_method_tester(tmp.diff, RuntimeError, None, None, None)
    abstract_method_tester(tmp.set_weights, NotImplementedError, None)
    abstract_method_tester(tmp.get_weights, NotImplementedError)
    try:
        tmp.weights_size
    except NotImplementedError:
        pass
    else:
        assert False
