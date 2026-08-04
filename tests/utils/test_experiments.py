from pytest import raises

from mushroom_rl.algorithms.value import DQN, DoubleDQN, Rainbow
from mushroom_rl.environments import SimpleChain
from mushroom_rl.utils import get_root_dir, get_log_dir, get_data_dir, select_class


def make_examples_tree(tmp_path):
    script = tmp_path / 'examples' / 'algorithms' / 'value' / 'experiment.py'
    script.parent.mkdir(parents=True)
    script.touch()

    return script


def test_get_root_dir(tmp_path):
    script = make_examples_tree(tmp_path)

    assert get_root_dir(script) == tmp_path / 'examples'
    assert get_root_dir(str(script)) == tmp_path / 'examples'
    assert get_root_dir(script.parent / 'other.py') == tmp_path / 'examples'


def test_get_root_dir_shallow(tmp_path):
    script = tmp_path / 'examples' / 'experiment.py'
    script.parent.mkdir(parents=True)
    script.touch()

    assert get_root_dir(script) == tmp_path / 'examples'


def test_get_root_dir_custom_name(tmp_path):
    script = tmp_path / 'benchmarks' / 'deep' / 'experiment.py'
    script.parent.mkdir(parents=True)
    script.touch()

    assert get_root_dir(script, root_name='benchmarks') == tmp_path / 'benchmarks'


def test_get_root_dir_missing(tmp_path):
    script = tmp_path / 'somewhere' / 'experiment.py'
    script.parent.mkdir(parents=True)
    script.touch()

    with raises(ValueError):
        get_root_dir(script)


def test_get_log_and_data_dir(tmp_path):
    script = make_examples_tree(tmp_path)

    assert get_log_dir(script) == tmp_path / 'examples' / 'logs'
    assert get_data_dir(script) == tmp_path / 'examples' / 'data'


def test_select_class():
    assert select_class('DQN', [DQN, DoubleDQN, Rainbow]) is DQN
    assert select_class('Rainbow', [DQN, DoubleDQN, Rainbow]) is Rainbow
    assert select_class('SimpleChain', [SimpleChain]) is SimpleChain


def test_select_class_unknown():
    with raises(ValueError) as error:
        select_class('SAC', [DQN, DoubleDQN])

    assert 'DQN' in str(error.value)
    assert 'DoubleDQN' in str(error.value)
