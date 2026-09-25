from mushroom_rl.core._impl.extra_info import StepInfo, EpisodeInfo
import pytest
import torch
import numpy as np


def test_list_of_dict():
    info = StepInfo(6, 'numpy')

    data = []
    for i in range(6):
        single_step_data = {
            'prop1': 100 + i,
            'prop2': np.arange(300 + i, 300 + i + 0.5, 0.1),
            'prop3': {
                'x': 400 + i,
                'y': 500 + i
            }
        }
        data.append(single_step_data)

    data2 = []
    for i in range(6):
        single_step_data = {
            'prop1': 110 + i,
            'prop2': np.arange(310 + i, 310 + i + 0.5, 0.1),
            'prop3': {
                'x': 410 + i,
                'y': 510 + i
            }
        }
        data2.append(single_step_data)

    info.append(data)
    info.append(data2)

    content = info.parse(to='torch')

    assert len(content) == 4
    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop2"])
    assert torch.is_tensor(content["prop3_x"])
    assert torch.is_tensor(content["prop3_y"])
    assert content["prop1"].dim() == 2 and content["prop1"].size(0) == 2 and content["prop1"].size(1) == 6
    prop2 = content["prop2"]
    assert prop2.dim() == 3 and prop2.size(0) == 2 and prop2.size(1) == 6 and prop2.size(2) == 5
    assert content["prop3_x"].dim() == 2 and content["prop3_x"].size(0) == 2 and content["prop3_x"].size(1) == 6
    assert content["prop3_y"].dim() == 2 and content["prop3_y"].size(0) == 2 and content["prop3_y"].size(1) == 6

    info = info.to_backend('torch')
    info = info.flatten()

    content = info.parse()
    assert len(content) == 4
    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop2"])
    assert torch.is_tensor(content["prop3_x"])
    assert torch.is_tensor(content["prop3_y"])
    assert content["prop1"].dim() == 1 and content["prop1"].size(0) == 12
    assert content["prop2"].dim() == 2 and content["prop2"].size(0) == 12 and content["prop2"].size(1) == 5
    assert content["prop3_x"].dim() == 1 and content["prop3_x"].size(0) == 12
    assert content["prop3_y"].dim() == 1 and content["prop3_y"].size(0) == 12

    prop1 = torch.tensor([100, 110, 101, 111, 102, 112, 103, 113, 104, 114, 105, 115])
    prop3_x = torch.tensor([400, 410, 401, 411, 402, 412, 403, 413, 404, 414, 405, 415])
    prop3_y = torch.tensor([500, 510, 501, 511, 502, 512, 503, 513, 504, 514, 505, 515])
    assert torch.equal(prop1, content["prop1"])
    assert torch.equal(prop3_x, content["prop3_x"])
    assert torch.equal(prop3_y, content["prop3_y"])

    content = info.parse(to='torch')

    assert len(content) == 4
    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop2"])
    assert torch.is_tensor(content["prop3_x"])
    assert torch.is_tensor(content["prop3_y"])
    assert content["prop1"].dim() == 1 and content["prop1"].size(0) == 12
    assert content["prop2"].dim() == 2 and content["prop2"].size(0) == 12 and content["prop2"].size(1) == 5
    assert content["prop3_x"].dim() == 1 and content["prop3_x"].size(0) == 12
    assert content["prop3_y"].dim() == 1 and content["prop3_y"].size(0) == 12


def test_dict_of_torch():
    info = StepInfo(4, 'torch')
    data1 = {
        'prop1': torch.arange(100, 104),
        'prop2': torch.tensor([[200.0, 200.5], [201.0, 201.5], [202.0, 202.5], [203.0, 203.5]]),
        'prop3': {
            'x': torch.arange(300, 304)
        }
    }
    data2 = {
        'prop1': torch.arange(110, 114),
        'prop2': torch.tensor([[210.0, 210.5], [211.0, 211.5], [212.0, 212.5], [213.0, 213.5]]),
        'prop3': {
            'x': torch.arange(310, 314)
        }
    }
    info.append(data1)
    info.append(data2)

    content = info.parse(to='numpy')

    assert len(content) == 3
    assert isinstance(content["prop1"], np.ndarray)
    assert isinstance(content["prop2"], np.ndarray)
    assert isinstance(content["prop3_x"], np.ndarray)
    assert content["prop1"].ndim == 2 and content["prop1"].shape[0] == 2 and content["prop1"].shape[1] == 4
    prop2 = content["prop2"]
    assert prop2.ndim == 3 and prop2.shape[0] == 2 and prop2.shape[1] == 4 and prop2.shape[2] == 2
    assert content["prop3_x"].ndim == 2 and content["prop3_x"].shape[0] == 2 and content["prop3_x"].shape[1] == 4

    info = info.to_backend('numpy')
    info = info.flatten()

    content = info.parse()
    assert len(content) == 3
    assert isinstance(content["prop1"], np.ndarray)
    assert isinstance(content["prop2"], np.ndarray)
    assert isinstance(content["prop3_x"], np.ndarray)
    assert content["prop1"].ndim == 1 and content["prop1"].shape[0] == 8
    assert content["prop2"].ndim == 2 and content["prop2"].shape[0] == 8 and content["prop2"].shape[1] == 2
    assert content["prop3_x"].ndim == 1 and content["prop3_x"].shape[0] == 8

    assert np.array_equal(np.array([100, 110, 101, 111, 102, 112, 103, 113]), content["prop1"])
    prop2 = np.array([[200.0, 200.5], [210.0, 210.5], [201.0, 201.5], [211.0, 211.5],
                      [202.0, 202.5], [212.0, 212.5], [203.0, 203.5], [213.0, 213.5]])
    assert np.array_equal(prop2, content["prop2"])
    assert np.array_equal(np.array([300, 310, 301, 311, 302, 312, 303, 313]), content["prop3_x"])

    content = info.parse()

    assert len(content) == 3
    assert isinstance(content["prop1"], np.ndarray)
    assert isinstance(content["prop2"], np.ndarray)
    assert isinstance(content["prop3_x"], np.ndarray)
    assert content["prop1"].ndim == 1 and content["prop1"].shape[0] == 8
    assert content["prop2"].ndim == 2 and content["prop2"].shape[0] == 8 and content["prop2"].shape[1] == 2
    assert content["prop3_x"].ndim == 1 and content["prop3_x"].shape[0] == 8


def test_empty_dict_in_list():
    info = StepInfo(3, 'torch')

    data1 = {
        'prop1': 100,
        'prop2': 200
    }
    data2 = {}
    data3 = {
        'prop1': 102,
        'prop2': 202
    }
    info.append([data1, data2, data3])
    info = info.flatten()
    print(info)
    content = info.parse()
    assert len(content) == 2

    assert "prop1" in content
    assert "prop2" in content

    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop2"])

    assert content["prop1"].dim() == 1 and content["prop1"].size(0) == 3
    assert content["prop2"].dim() == 1 and content["prop2"].size(0) == 3

    assert content["prop1"][0] == 100 and content["prop2"][0] == 200
    assert torch.isnan(content["prop1"][1]) and torch.isnan(content["prop2"][1])
    assert content["prop1"][2] == 102 and content["prop2"][2] == 202


def test_empty_dict():
    info = StepInfo(2, 'numpy')
    data1 = {
        'prop1': np.arange(100, 102)
    }
    data2 = {}
    data3 = {
        'prop1': np.arange(120, 122)
    }
    info.append(data1)
    info.append(data2)
    info.append(data3)
    info = info.flatten()
    print(info)

    content = info.parse()
    assert len(content) == 1
    assert "prop1" in content
    assert isinstance(content["prop1"], np.ndarray)
    assert content["prop1"].ndim == 1 and content["prop1"].shape[0] == 6

    assert content["prop1"][0] == 100
    assert np.isnan(content["prop1"][1])
    assert content["prop1"][2] == 120
    assert content["prop1"][3] == 101
    assert np.isnan(content["prop1"][4])
    assert content["prop1"][5] == 121


def test_changing_properties_dict():
    info = StepInfo(2, 'numpy')
    data1 = {
        'prop2': np.arange(200, 202),
        'prop3': np.arange(300, 302)
    }
    data2 = {
        'prop2': np.arange(210, 212),
        'prop4': np.arange(410, 412)
    }
    data3 = {
        'prop2': np.arange(220, 222),
        'prop3': np.arange(320, 322)
    }
    info.append(data1)
    info.append(data2)
    info.append(data3)
    info = info.to_backend('torch')
    info = info.flatten()

    print(info)

    content = info.parse()
    assert len(content) == 3

    assert "prop2" in content
    assert "prop3" in content
    assert "prop4" in content

    assert torch.is_tensor(content["prop2"])
    assert torch.is_tensor(content["prop3"])
    assert torch.is_tensor(content["prop4"])

    assert content["prop2"].dim() == 1 and content["prop2"].size(0) == 6
    assert content["prop3"].dim() == 1 and content["prop3"].size(0) == 6
    assert content["prop4"].dim() == 1 and content["prop4"].size(0) == 6

    assert content["prop2"][0] == 200 and content["prop3"][0] == 300 and torch.isnan(content["prop4"][0])
    assert content["prop2"][1] == 210 and torch.isnan(content["prop3"][1]) and content["prop4"][1] == 410
    assert content["prop2"][2] == 220 and content["prop3"][2] == 320 and torch.isnan(content["prop4"][2])
    assert content["prop2"][3] == 201 and content["prop3"][3] == 301 and torch.isnan(content["prop4"][3])
    assert content["prop2"][4] == 211 and torch.isnan(content["prop3"][4]) and content["prop4"][4] == 411
    assert content["prop2"][5] == 221 and content["prop3"][5] == 321 and torch.isnan(content["prop4"][5])


def test_one_environment():
    info = StepInfo(1, 'torch')
    data1 = {
        'prop1': torch.arange(100, 103),
        'prop2': torch.randn(3, 2),
        'prop3': 1
    }
    data2 = {
        'prop1': torch.arange(110, 113),
        'prop2': torch.randn(3, 2),
        'prop3': 2
    }
    data3 = {
        'prop1': torch.arange(120, 123),
        'prop2': torch.randn(3, 2),
        'prop3': 3
    }
    info.append(data1)
    info.append(data2)
    info.append(data3)
    content = info.parse('torch')
    print(info)

    assert len(content) == 3

    assert "prop1" in content
    assert "prop2" in content
    assert "prop3" in content

    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop2"])
    assert torch.is_tensor(content["prop3"])

    assert content["prop1"].dim() == 2 and content["prop1"].size(0) == 3 and content["prop2"].size(1) == 3
    prop2 = content["prop2"]
    assert prop2.dim() == 3 and prop2.size(0) == 3 and prop2.size(1) == 3 and prop2.size(2) == 2
    assert content["prop3"].dim() == 1 and content["prop3"].size(0) == 3


def test_copy_survives_parse():
    info = StepInfo(1, 'numpy')

    for i in range(4):
        info.append({'a': 10 + i, 'b': {'x': 20 + i}})

    info.parse()

    copied = info.copy()

    content = copied.parse()
    assert np.array_equal(content['a'], np.array([10, 11, 12, 13]))
    assert np.array_equal(content['b_x'], np.array([20, 21, 22, 23]))

    content = copied.parse()

    assert np.array_equal(content['a'], np.array([10, 11, 12, 13]))
    assert np.array_equal(content['b_x'], np.array([20, 21, 22, 23]))

    view = copied.get_view(slice(1, 3))

    content = view.parse()
    assert np.array_equal(content['a'], np.array([11, 12]))
    assert np.array_equal(content['b_x'], np.array([21, 22]))


def test_merge_keeps_the_row_order():
    unparsed = StepInfo(1, 'numpy')
    for i in range(4):
        unparsed.append({'x': float(i)})

    parsed = StepInfo(1, 'numpy')
    for i in (4, 5):
        parsed.append({'x': float(i)})
    parsed.parse()

    merged = unparsed + parsed
    content = merged.parse()

    assert np.array_equal(content['x'], np.array([0., 1., 2., 3., 4., 5.]))

    unparsed += parsed
    content = unparsed.parse()

    assert np.array_equal(content['x'], np.array([0., 1., 2., 3., 4., 5.]))


def test_merge_keeps_the_destination_backend():
    torch_info = StepInfo(1, 'torch')

    for value in (1., 2.):
        numpy_info = StepInfo(1, 'numpy')
        numpy_info.append({'x': value})
        numpy_info.parse()
        torch_info += numpy_info

    content = torch_info.parse()
    assert torch.equal(content['x'], torch.tensor([1., 2.]))

    content = torch_info.parse()

    assert torch.equal(content['x'], torch.tensor([1., 2.]))


def test_get_view_slice():
    info = StepInfo(3, 'torch')
    data1 = {
        'prop1': torch.arange(100, 103),
        'prop3': torch.randn(3, 2)
    }
    data2 = {
        'prop1': torch.arange(110, 113),
        'prop3': torch.randn(3, 2)
    }

    info.append(data1)
    info.append(data2)

    info = info.flatten()
    info = info.get_view(slice(4))
    content = info.parse('torch')

    assert len(content) == 2

    assert "prop1" in content
    assert "prop3" in content

    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop3"])

    assert content["prop1"].dim() == 1 and content["prop1"].size(0) == 4
    assert content["prop3"].dim() == 2 and content["prop3"].size(0) == 4 and content["prop3"].size(1) == 2

    assert content["prop1"][0] == 100
    assert content["prop1"][1] == 110
    assert content["prop1"][2] == 101
    assert content["prop1"][3] == 111


def test_get_view_array():
    info = StepInfo(3, 'torch')
    data1 = {
        'prop1': torch.arange(100, 103),
        'prop3': torch.randn(3, 2)
    }
    data2 = {
        'prop1': torch.arange(110, 113),
        'prop3': torch.randn(3, 2)
    }

    info.append(data1)
    info.append(data2)

    info = info.flatten()
    info = info.get_view(np.array([1, 2, 5]), True)
    content = info.parse('torch')
    print(info)

    assert len(content) == 2

    assert "prop1" in content
    assert "prop3" in content

    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop3"])

    assert content["prop1"].dim() == 1 and content["prop1"].size(0) == 3
    assert content["prop3"].dim() == 2 and content["prop3"].size(0) == 3 and content["prop3"].size(1) == 2

    assert content["prop1"][0] == 110
    assert content["prop1"][1] == 101
    assert content["prop1"][2] == 112


def test_add():
    info1 = StepInfo(10, 'numpy')
    data1 = {
        'prop1': np.arange(100, 110),
        'prop2': np.arange(200, 210)
    }
    data2 = {
        'prop1': np.arange(110, 120),
        'prop2': np.arange(210, 220)
    }
    info1.append(data1)
    info1.append(data2)

    info2 = StepInfo(10, 'torch')
    data1 = {
        'prop1': torch.arange(100, 110, dtype=torch.float32),
        'prop3': torch.arange(300, 310, dtype=torch.float32)
    }
    data2 = {
        'prop1': torch.arange(110, 120),
        'prop3': torch.arange(310, 320)
    }
    info2.append(data1)
    info2.append(data2)

    info1 = info1.to_backend('torch')
    info2 = info2.to_backend('numpy')

    info = info1 + info2

    content = info.parse()
    assert len(content) == 3

    assert "prop1" in content
    assert "prop2" in content
    assert "prop3" in content

    assert torch.is_tensor(content["prop1"])
    assert torch.is_tensor(content["prop2"])
    assert torch.is_tensor(content["prop3"])

    assert content["prop1"].dim() == 2 and content["prop1"].size(0) == 4 and content["prop1"].size(1) == 10
    assert content["prop2"].dim() == 2 and content["prop2"].size(0) == 4 and content["prop2"].size(1) == 10
    assert content["prop3"].dim() == 2 and content["prop3"].size(0) == 4 and content["prop3"].size(1) == 10

    for i in range(2):
        for j in range(10):
            assert content["prop1"][i][j] == 100 + i*10 + j
            assert content["prop2"][i][j] == 200 + i*10 + j
            assert torch.isnan(content["prop3"][i][j])

    for i in range(2):
        for j in range(10):
            assert content["prop1"][2 + i][j] == 100 + i*10 + j
            assert torch.isnan(content["prop2"][2 + i][j])
            assert content["prop3"][2 + i][j] == 300 + i*10 + j


def test_clear():
    info = StepInfo(10, 'numpy')
    data1 = {
        'prop1': np.arange(100, 110),
        'prop2': np.arange(200, 210)
    }
    data2 = {
        'prop1': np.arange(110, 120),
        'prop2': np.arange(210, 220)
    }
    info.append(data1)
    info.append(data2)
    info.parse()
    info.clear()
    content = info.parse()
    assert not content


def test_flatten_with_mask():
    info = StepInfo(5, 'numpy')
    data1 = {
        'prop1': np.arange(100, 105),
        'prop2': np.arange(200, 205)
    }
    data2 = {
        'prop1': np.arange(110, 115),
        'prop2': np.arange(210, 215)
    }
    info.append(data1)
    info.append(data2)
    mask = np.array([True, True, False, False, False, True, False, False, True, False])
    info = info.flatten(mask)

    content = info.parse()
    assert len(content) == 2

    assert "prop1" in content
    assert "prop2" in content

    assert isinstance(content["prop1"], np.ndarray)
    assert isinstance(content["prop2"], np.ndarray)

    assert content["prop1"].ndim == 1 and content["prop1"].shape[0] == 4
    assert content["prop2"].ndim == 1 and content["prop2"].shape[0] == 4

    assert np.array_equal(np.array([100, 110, 112, 104]), content["prop1"])
    assert np.array_equal(np.array([200, 210, 212, 204]), content["prop2"])

    # Test if mask is permantly applied
    content = info.parse()
    assert len(content) == 2

    assert "prop1" in content
    assert "prop2" in content

    assert isinstance(content["prop1"], np.ndarray)
    assert isinstance(content["prop2"], np.ndarray)

    assert content["prop1"].ndim == 1 and content["prop1"].shape[0] == 4
    assert content["prop2"].ndim == 1 and content["prop2"].shape[0] == 4

    assert np.array_equal(np.array([100, 110, 112, 104]), content["prop1"])
    assert np.array_equal(np.array([200, 210, 212, 204]), content["prop2"])


def test_list_backend_sequential():
    info = StepInfo(1, 'list')

    info.append({'a': 1.0, 'nested': {'v': np.array([1.0, 2.0])}})
    info.append({'a': 2.0, 'nested': {'v': np.array([3.0, 4.0])}})
    info.append({'a': 3.0, 'nested': {'v': np.array([5.0, 6.0])}})

    content = info.parse()

    assert isinstance(content['a'], list)
    assert content['a'] == [1.0, 2.0, 3.0]
    assert isinstance(content['nested_v'], list)
    assert np.array_equal(content['nested_v'][0], np.array([1.0, 2.0]))
    assert np.array_equal(content['nested_v'][2], np.array([5.0, 6.0]))

    view = info.get_view(slice(0, 2))
    content = view.parse()
    assert content['a'] == [1.0, 2.0]

    view = info.get_view(np.array([2, 0]))
    content = view.parse()
    assert content['a'] == [3.0, 1.0]


def test_list_backend_vectorized():
    info = StepInfo(2, 'list', vectorized=True)

    info.append([{'a': 1.0}, {'a': 2.0}])
    info.append([{'a': 3.0}, {'a': 4.0}])

    content = info.parse()

    assert content['a'] == [[1.0, 2.0], [3.0, 4.0]]

    flat = info.flatten()
    content = flat.parse()
    assert content['a'] == [1.0, 3.0, 2.0, 4.0]

    mask = np.array([[True, True], [True, False]])
    flat_masked = info.flatten(mask)
    content = flat_masked.parse()
    assert content['a'] == [1.0, 3.0, 2.0]


def test_records_layout_defers_everything():
    info = StepInfo(2, 'numpy')

    for t in range(3):
        info.append([{'a': float(10 * t + j)} for j in range(2)])

    assert info._layout == 'records'
    assert len(info._records) == 3
    assert not info._columns

    content = info.parse()

    assert np.array_equal(content['a'], np.array([[0., 1.], [10., 11.], [20., 21.]]))


def test_columns_layout_stores_one_entry_per_key():
    info = StepInfo(2, 'numpy')

    for t in range(3):
        info.append({'a': np.array([10. * t, 10. * t + 1]), 'nested': {'x': np.array([float(t), float(t)])}})

    assert info._layout == 'columns'
    assert not info._records
    assert sorted(info._columns.keys()) == ['a', 'nested_x']
    assert info._column_rows['a'] == [0, 1, 2]

    content = info.parse()

    assert np.array_equal(content['a'], np.array([[0., 1.], [10., 11.], [20., 21.]]))
    assert np.array_equal(content['nested_x'], np.array([[0., 0.], [1., 1.], [2., 2.]]))


def test_columns_layout_pads_the_missing_steps():
    info = StepInfo(2, 'numpy')

    info.append({'a': np.array([1., 2.])})
    info.append({'a': np.array([3., 4.]), 'rare': np.array([5., 6.])})
    info.append({'a': np.array([7., 8.])})

    assert info._column_rows['rare'] == [1]

    content = info.parse()

    assert np.array_equal(content['a'], np.array([[1., 2.], [3., 4.], [7., 8.]]))
    assert np.all(np.isnan(content['rare'][0]))
    assert np.array_equal(content['rare'][1], np.array([5., 6.]))
    assert np.all(np.isnan(content['rare'][2]))


def test_parse_is_cached_until_an_append():
    info = StepInfo(1, 'numpy')

    info.append({'a': 1.})
    first = info.parse()

    assert info.parse() is first

    info.append({'a': 2.})
    second = info.parse()

    assert second is not first
    assert np.array_equal(first['a'], np.array([1.]))
    assert np.array_equal(second['a'], np.array([1., 2.]))


def test_flatten_is_deferred_and_snapshots_the_source():
    info = StepInfo(2, 'numpy')

    for t in range(3):
        info.append([{'a': float(10 * t + j)} for j in range(2)])

    flat = info.flatten()
    copied = flat.copy()

    assert flat._source is not None

    info.append([{'a': 99.}, {'a': 98.}])
    info.clear()

    assert np.array_equal(flat.parse()['a'], np.array([0., 10., 20., 1., 11., 21.]))
    assert np.array_equal(copied.parse()['a'], np.array([0., 10., 20., 1., 11., 21.]))
    assert flat._source is None


def test_key_appearing_after_a_parse():
    info = StepInfo(1, 'numpy')

    info.append({'x': 1.})
    info.parse()
    info.append({'x': 2., 'y': 3.})

    content = info.parse()

    assert np.array_equal(content['x'], np.array([1., 2.]))
    assert np.isnan(content['y'][0])
    assert content['y'][1] == 3.


def test_save_and_load(tmpdir):
    info = StepInfo(3, 'numpy')

    info.append([{'a': float(j), 'nested': {'b': float(10 + j)}} for j in range(3)])
    info.append([{'a': float(100 + j), 'nested': {'b': float(110 + j)}} for j in range(3)])

    path = str(tmpdir / 'info.msh')
    info.save(path)
    loaded = StepInfo.load(path)

    assert loaded.n_envs == 3
    assert np.array_equal(loaded.parse()['a'], np.array([[0., 1., 2.], [100., 101., 102.]]))
    assert np.array_equal(loaded.parse()['nested_b'], np.array([[10., 11., 12.], [110., 111., 112.]]))
    assert np.array_equal(loaded.copy().parse()['a'], np.array([[0., 1., 2.], [100., 101., 102.]]))
    assert np.array_equal(loaded.flatten().parse()['a'], np.array([0., 100., 1., 101., 2., 102.]))


def test_episode_info_keeps_one_entry_per_episode_and_environment():
    info = EpisodeInfo(3, 'numpy')

    info.append([{'ep': float(j)} for j in range(3)], np.ones(3, dtype=bool))
    info.append({'ep': np.array([10., 11., 12.])}, np.array([True, False, True]))

    assert len(info) == 5

    content = info.parse()

    assert np.array_equal(content['ep'], np.array([0., 10., 1., 2., 12.]))


def test_episode_info_of_a_single_environment_takes_the_dictionary_as_is():
    info = EpisodeInfo(1, 'numpy')

    info.append({'ep': 1.})
    info.append({'ep': 2.})

    content = info.parse()

    assert np.array_equal(content['ep'], np.array([1., 2.]))


def test_episode_info_flatten_concatenates_the_environments():
    info = EpisodeInfo(2, 'numpy')

    info.append([{'ep': 0.}, {'ep': 1.}], np.ones(2, dtype=bool))
    info.append([{'ep': 2.}, {'ep': 3.}], np.array([True, False]))

    flat = info.flatten()

    assert flat.n_envs == 1
    assert np.array_equal(flat.parse()['ep'], np.array([0., 2., 1.]))


def test_episode_info_keeps_raw_entries_apart():
    info = EpisodeInfo(3, 'numpy')

    info.append(np.arange(3.), np.array([True, False, True]))
    info.append(np.arange(3.) + 10, np.array([True, True, False]))

    assert info.n_envs == 3
    assert len(info) == 4
    assert info.episodes == [[0., 10.], [11.], [2.]]
    assert info.flatten().episodes == [0., 10., 11., 2.]


def test_episode_info_of_a_single_environment_takes_the_entry_as_is():
    info = EpisodeInfo(1, 'numpy')

    info.append(1.)
    info.append(2.)

    assert info.n_envs == 1
    assert info.episodes == [1., 2.]
    assert info.flatten().episodes == [1., 2.]


def test_episode_info_merges_environment_by_environment():
    first = EpisodeInfo(2, 'numpy')
    first.append([0., 1.], np.ones(2, dtype=bool))

    second = EpisodeInfo(2, 'numpy')
    second.append([2., 3.], np.ones(2, dtype=bool))
    second.append([4., 5.], np.array([False, True]))

    merged = first + second

    assert merged.episodes == [[0., 2.], [1., 3., 5.]]
    assert first.episodes == [[0.], [1.]]

    first += second

    assert first.episodes == [[0., 2.], [1., 3., 5.]]


def test_episode_info_copy_and_clear_are_independent():
    info = EpisodeInfo(2, 'numpy')
    info.append([1., 2.], np.ones(2, dtype=bool))

    copied = info.copy()
    empty = info.empty()
    info.clear()

    assert copied.episodes == [[1.], [2.]]
    assert empty.episodes == [[], []]
    assert empty.n_envs == 2
    assert info.episodes == [[], []]


def test_merge_keeps_the_row_order_for_every_parsed_combination():
    for own_parsed in (False, True):
        for other_parsed in (False, True):
            first = StepInfo(1, 'numpy')
            for i in range(3):
                first.append({'x': float(i)})
            if own_parsed:
                first.parse()

            second = StepInfo(1, 'numpy')
            for i in (3, 4):
                second.append({'x': float(i)})
            if other_parsed:
                second.parse()

            merged = first + second

            assert np.array_equal(merged.parse()['x'], np.array([0., 1., 2., 3., 4.]))
            assert np.array_equal(first.parse()['x'], np.array([0., 1., 2.]))

            first += second

            assert np.array_equal(first.parse()['x'], np.array([0., 1., 2., 3., 4.]))


def test_merge_keeps_the_row_order_for_the_columns_layout():
    for own_parsed in (False, True):
        for other_parsed in (False, True):
            first = StepInfo(2, 'numpy')
            for i in range(3):
                first.append({'x': np.array([float(i), float(i) + 0.5])})
            if own_parsed:
                first.parse()

            second = StepInfo(2, 'numpy')
            for i in (3, 4):
                second.append({'x': np.array([float(i), float(i) + 0.5])})
            if other_parsed:
                second.parse()

            first += second

            assert np.array_equal(first.parse()['x'],
                                  np.array([[0., 0.5], [1., 1.5], [2., 2.5], [3., 3.5], [4., 4.5]]))


def test_merge_of_an_unparsed_side_stays_unparsed():
    first = StepInfo(1, 'numpy')
    first.append({'x': 1.})

    second = StepInfo(1, 'numpy')
    second.append({'x': 2.})

    first += second

    assert first._pending_steps == 2
    assert not first._parsed
    assert np.array_equal(first.parse()['x'], np.array([1., 2.]))


def test_merge_with_a_missing_key_on_one_side():
    first = StepInfo(1, 'numpy')
    first.append({'x': 1.})

    second = StepInfo(1, 'numpy')
    second.append({'x': 2., 'y': 3.})

    first += second
    content = first.parse()

    assert np.array_equal(content['x'], np.array([1., 2.]))
    assert np.isnan(content['y'][0])
    assert content['y'][1] == 3.


def test_drop_before_keeps_the_remaining_steps_unparsed():
    for layout in ('records', 'columns'):
        info = StepInfo(2, 'numpy')
        for t in range(4):
            if layout == 'records':
                info.append([{'x': float(10 * t + j)} for j in range(2)])
            else:
                info.append({'x': np.array([float(10 * t), float(10 * t + 1)])})

        info.drop_before(2)

        assert not info._parsed
        assert info._pending_steps == 2
        assert np.array_equal(info.parse()['x'], np.array([[20., 21.], [30., 31.]]))


def test_drop_before_slices_the_already_parsed_steps():
    info = StepInfo(1, 'numpy')
    for t in range(4):
        info.append({'x': float(t)})
    info.parse()
    info.append({'x': 4.})

    info.drop_before(2)

    assert np.array_equal(info.parse()['x'], np.array([2., 3., 4.]))


def test_drop_before_forgets_a_key_left_without_steps():
    for layout in ('records', 'columns'):
        info = StepInfo(2, 'numpy')
        for t in range(4):
            entry = {'x': float(t)} if layout == 'records' else {'x': np.array([float(t), float(t)])}
            if t == 0:
                if layout == 'records':
                    entry['rare'] = 1.
                else:
                    entry['rare'] = np.array([1., 1.])
            info.append([entry, entry] if layout == 'records' else entry)

        info.drop_before(2)

        assert 'rare' not in info.parse()


def test_a_key_forgotten_by_drop_before_comes_back_padded():
    info = StepInfo(1, 'numpy')
    info.append({'x': 0., 'rare': 9.})
    info.append({'x': 1.})
    info.drop_before(1)

    info.append({'x': 2., 'rare': 8.})
    content = info.parse()

    assert np.array_equal(content['x'], np.array([1., 2.]))
    assert np.isnan(content['rare'][0])
    assert content['rare'][1] == 8.


def test_merging_after_drop_before_pads_the_missing_key():
    dropped = StepInfo(1, 'numpy')
    dropped.append({'x': 0., 'rare': 9.})
    dropped.append({'x': 1.})
    dropped.drop_before(1)

    other = StepInfo(1, 'numpy')
    other.append({'x': 2., 'rare': 8.})
    other.parse()

    dropped += other
    content = dropped.parse()

    assert np.array_equal(content['x'], np.array([1., 2.]))
    assert np.isnan(content['rare'][0])
    assert content['rare'][1] == 8.


def test_merging_an_unparsed_flattened_info_keeps_its_rows():
    first = StepInfo(2, 'numpy')
    first.append({'x': np.array([0., 1.])})
    first.append({'x': np.array([2., 3.])})

    second = StepInfo(2, 'numpy')
    second.append({'x': np.array([10., 11.])})
    second.append({'x': np.array([12., 13.])})

    flat = first.flatten()
    flat += second.flatten()

    assert np.array_equal(flat.parse()['x'], np.array([0., 2., 1., 3., 10., 12., 11., 13.]))


def test_merging_an_unparsed_flattened_info_keeps_its_rows_when_parsed_first():
    first = StepInfo(2, 'numpy')
    first.append({'x': np.array([0., 1.])})

    second = StepInfo(2, 'numpy')
    second.append({'x': np.array([10., 11.])})

    flat = first.flatten()
    flat.parse()
    flat += second.flatten()

    assert np.array_equal(flat.parse()['x'], np.array([0., 1., 10., 11.]))


def test_a_key_merged_into_a_flattened_info_survives_the_resolution():
    info = StepInfo(2, 'numpy')
    info.append({'t': np.array([1., 2.])})

    other = StepInfo(1, 'numpy', vectorized=False)
    other.append({'t': 5., 'final': 7.})

    flat = info.flatten()
    flat += other
    content = flat.parse()

    assert np.array_equal(content['t'], np.array([1., 2., 5.]))
    assert np.isnan(content['final'][0])
    assert np.isnan(content['final'][1])
    assert content['final'][2] == 7.


def test_a_key_appended_to_a_flattened_info_survives_the_resolution():
    info = StepInfo(2, 'numpy')
    info.append({'t': np.array([1., 2.])})

    flat = info.flatten()
    flat.append({'t': 5., 'final': 7.})
    content = flat.parse()

    assert np.array_equal(content['t'], np.array([1., 2., 5.]))
    assert np.isnan(content['final'][0])
    assert np.isnan(content['final'][1])
    assert content['final'][2] == 7.


def test_drop_before_forgets_a_key_of_a_copy():
    info = StepInfo(1, 'numpy')
    info.append({'x': 1.})
    info.append({'y': 2.})

    duplicate = info.copy()
    duplicate.drop_before(1)

    assert np.array_equal(duplicate.parse()['y'], np.array([2.]))
    assert 'x' not in duplicate.parse()


def test_drop_before_forgets_a_key_taken_from_a_merge():
    info = StepInfo(1, 'numpy')
    info.append({'x': 1.})

    other = StepInfo(1, 'numpy')
    other.append({'x': 2., 'y': 9.})
    other.append({'x': 3.})

    info += other
    info.drop_before(2)

    assert np.array_equal(info.parse()['x'], np.array([3.]))
    assert 'y' not in info.parse()


def test_parse_to_another_backend_drops_the_device():
    info = StepInfo(1, 'torch', device='cpu')
    info.append({'x': 1.})

    content = info.parse(to='numpy')

    assert np.array_equal(content['x'], np.array([1.]))


def test_merging_a_missing_key_keeps_the_device():
    info = StepInfo(1, 'torch', device='meta')
    info.append({'x': torch.tensor(1.)})
    info.parse()

    other = StepInfo(1, 'torch', device='meta')
    other.append({'x': torch.tensor(2.), 'extra': torch.tensor(3.)})
    other.parse()

    info += other
    content = info.parse()

    assert content['x'].device.type == 'meta'
    assert content['extra'].device.type == 'meta'


def test_drop_before_refuses_an_unresolved_flattened_info():
    info = StepInfo(2, 'numpy')
    info.append({'x': np.array([1., 2.])})

    flat = info.flatten()

    with pytest.raises(AssertionError):
        flat.drop_before(1)


def test_a_parse_without_keys_keeps_the_step_count():
    info = StepInfo(1, 'numpy')
    for _ in range(5):
        info.append({})

    info.parse()

    for t in range(3):
        info.append({'episode_r': float(t)})

    content = info.parse()

    assert content['episode_r'].shape == (8,)
    assert np.isnan(content['episode_r'][:5]).all()
    assert np.array_equal(content['episode_r'][5:], np.array([0., 1., 2.]))


def test_drop_before_after_a_parse_without_keys():
    info = StepInfo(1, 'numpy')
    for _ in range(4):
        info.append({})

    info.parse()

    info.append({'k': 7.})
    info.append({'k': 8.})
    info.drop_before(2)

    content = info.parse()

    assert content['k'].shape == (4,)
    assert np.array_equal(content['k'][2:], np.array([7., 8.]))


def test_merging_an_info_without_keys_keeps_its_steps():
    keyless = StepInfo(1, 'numpy')
    for _ in range(3):
        keyless.append({})
    keyless.parse()

    other = StepInfo(1, 'numpy')
    other.append({'k': 5.})
    other.parse()

    keyless += other
    content = keyless.parse()

    assert content['k'].shape == (4,)
    assert content['k'][3] == 5.


def test_to_backend_converts_the_parsed_arrays():
    info = StepInfo(1, 'torch')
    info.append({'x': torch.tensor(1.)})

    converted = info.to_backend('numpy')

    assert isinstance(converted.parse()['x'], np.ndarray)
    assert isinstance(info.parse()['x'], torch.Tensor)


def test_to_backend_defers_the_conversion():
    info = StepInfo(1, 'numpy')
    info.append({'x': 1.})

    converted = info.to_backend('torch')

    assert converted._parsed == {}
    assert converted._pending_steps == 1
    assert isinstance(converted.parse()['x'], torch.Tensor)
    assert isinstance(info.parse()['x'], np.ndarray)


def test_to_backend_converts_the_episode_info():
    info = EpisodeInfo(1, 'numpy')
    info.append({'ep': np.array([3.])})

    converted = info.to_backend('torch')

    assert isinstance(converted.parse()['ep'], torch.Tensor)
    assert isinstance(info.parse()['ep'], np.ndarray)


def test_flatten_after_to_backend():
    info = StepInfo(2, 'numpy')
    info.append({'x': np.array([1., 2.])})
    info.append({'x': np.array([3., 4.])})

    flat = info.to_backend('torch').flatten()
    content = flat.parse()

    assert isinstance(content['x'], torch.Tensor)
    assert torch.equal(content['x'], torch.tensor([1., 3., 2., 4.]))
