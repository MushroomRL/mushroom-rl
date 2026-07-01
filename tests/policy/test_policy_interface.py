from mushroom_rl.policy import Policy, HasWeights, HasGradient


def abstract_method_tester(f, ex, *args):
    try:
        f(*args)
    except ex:
        pass
    else:
        assert False


def test_policy_interface():
    tmp = Policy()
    abstract_method_tester(tmp.__call__, NotImplementedError, None, None)
    abstract_method_tester(tmp.draw_action, NotImplementedError, None)
    assert tmp.reset() is None
    assert tmp.reset_vectorized(None) is None
    assert not tmp.is_stateful


def test_has_weights():
    tmp = HasWeights()
    abstract_method_tester(tmp.set_weights, NotImplementedError, None)
    abstract_method_tester(tmp.get_weights, NotImplementedError)
    try:
        tmp.weights_size
    except NotImplementedError:
        pass
    else:
        assert False


def test_has_gradient():
    tmp = HasGradient()
    abstract_method_tester(tmp.diff_log, NotImplementedError, None, None)


def test_mixin_requires_policy():
    try:
        class BadPolicy(HasWeights):
            pass
    except TypeError:
        pass
    else:
        assert False
