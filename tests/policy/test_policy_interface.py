from mushroom.policy import Policy, ParametricPolicy


def abstract_method_tester(f, ex, *args):
    try:
        f(*args)
    except ex:
        pass
    else:
        assert False


def test_policy_interface():
    tmp = Policy()
    abstract_method_tester(tmp.__call__, NotImplementedError)
    abstract_method_tester(tmp.draw_action, NotImplementedError, None)
    tmp.reset()


def test_parametric_policy():
    tmp = ParametricPolicy()
    abstract_method_tester(tmp.)