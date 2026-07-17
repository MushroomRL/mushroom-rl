import pytest

from mushroom_rl.policy import Policy, HasWeights, HasGradient, StatefulPolicy


def abstract_method_tester(f, ex, *args):
    with pytest.raises(ex):
        f(*args)


def test_policy_interface():
    tmp = Policy()
    abstract_method_tester(tmp.__call__, NotImplementedError, None, None)
    abstract_method_tester(tmp.draw_action, NotImplementedError, None)
    abstract_method_tester(tmp.draw_action_greedy, NotImplementedError, None)
    assert tmp.reset() is None
    assert tmp.reset_vectorized(None) is None
    assert not tmp.is_stateful


def test_stateful_policy_interface():
    tmp = StatefulPolicy(policy_state_shape=(1,))
    assert tmp.is_stateful

    abstract_method_tester(tmp.reset, NotImplementedError)
    abstract_method_tester(tmp.reset_vectorized, NotImplementedError, None)
    abstract_method_tester(tmp.draw_action, NotImplementedError, None)
    abstract_method_tester(tmp.draw_action_greedy, NotImplementedError, None)


def test_has_weights():
    tmp = HasWeights()
    abstract_method_tester(tmp.set_weights, NotImplementedError, None)
    abstract_method_tester(tmp.get_weights, NotImplementedError)
    with pytest.raises(NotImplementedError):
        tmp.weights_size


def test_has_gradient():
    tmp = HasGradient()
    abstract_method_tester(tmp.diff_log, NotImplementedError, None, None)


def test_mixin_requires_policy():
    with pytest.raises(TypeError):
        class BadPolicy(HasWeights):
            pass


def test_mixin_requires_mixin_first():
    with pytest.raises(TypeError):
        class MixinLast(Policy, HasWeights):
            pass

    with pytest.raises(TypeError):
        class GradientMixinLast(Policy, HasGradient):
            pass


def test_mixin_accepts_mixin_first():
    class GoodPolicy(HasWeights, Policy):
        pass

    class GoodGradientPolicy(HasGradient, Policy):
        pass

    assert issubclass(GoodPolicy, Policy)
    assert issubclass(GoodGradientPolicy, HasWeights)


def test_intermediate_mixin_allowed():
    class IntermediateMixin(HasWeights, is_mixin=True):
        pass

    assert not issubclass(IntermediateMixin, Policy)
