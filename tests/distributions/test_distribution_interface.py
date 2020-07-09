from mushroom_rl.distributions import Distribution


def abstract_method_tester(f, *args):
    try:
        f(*args)
    except NotImplementedError:
        pass
    else:
        assert False


def test_distribution_interface():
    tmp = Distribution()

    abstract_method_tester(tmp.sample)
    abstract_method_tester(tmp.log_pdf, None)
    abstract_method_tester(tmp.__call__, None)
    abstract_method_tester(tmp.entropy)
    abstract_method_tester(tmp.mle, None)
    abstract_method_tester(tmp.diff_log, None)
    abstract_method_tester(tmp.diff, None)

    abstract_method_tester(tmp.get_parameters)
    abstract_method_tester(tmp.set_parameters, None)

    try:
        tmp.parameters_size
    except NotImplementedError:
        pass
    else:
        assert False
