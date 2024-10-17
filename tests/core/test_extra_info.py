from mushroom_rl.core import ExtraInfo
import torch
import numpy as np

def test_list_of_dict():
    info = ExtraInfo(6, 'numpy')

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

    info.parse(to='torch')

    assert(len(info) == 4)
    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop2"]))
    assert(torch.is_tensor(info["prop3_x"]))
    assert(torch.is_tensor(info["prop3_y"]))
    assert(info["prop1"].dim() == 2 and info["prop1"].size(0) == 2 and info["prop1"].size(1) == 6)
    assert(info["prop2"].dim() == 3 and info["prop2"].size(0) == 2 and info["prop2"].size(1) == 6 and info["prop2"].size(2) == 5)
    assert(info["prop3_x"].dim() == 2 and info["prop3_x"].size(0) == 2 and info["prop3_x"].size(1) == 6)
    assert(info["prop3_y"].dim() == 2 and info["prop3_y"].size(0) == 2 and info["prop3_y"].size(1) == 6)

    info = info.flatten()

    assert(len(info) == 4)
    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop2"]))
    assert(torch.is_tensor(info["prop3_x"]))
    assert(torch.is_tensor(info["prop3_y"]))
    assert(info["prop1"].dim() == 1 and info["prop1"].size(0) == 12)
    assert(info["prop2"].dim() == 2 and info["prop2"].size(0) == 12 and info["prop2"].size(1) == 5)
    assert(info["prop3_x"].dim() == 1 and info["prop3_x"].size(0) == 12)
    assert(info["prop3_y"].dim() == 1 and info["prop3_y"].size(0) == 12)

    prop1 = torch.tensor([100, 110, 101, 111, 102, 112, 103, 113, 104, 114, 105, 115])
    prop3_x = torch.tensor([400, 410, 401, 411, 402, 412, 403, 413, 404, 414, 405, 415])
    prop3_y = torch.tensor([500, 510, 501, 511, 502, 512, 503, 513, 504, 514, 505, 515])
    assert torch.equal(prop1, info["prop1"])
    assert torch.equal(prop3_x, info["prop3_x"])
    assert torch.equal(prop3_y, info["prop3_y"])

    info.parse(to='torch')

    assert(len(info) == 4)
    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop2"]))
    assert(torch.is_tensor(info["prop3_x"]))
    assert(torch.is_tensor(info["prop3_y"]))
    assert(info["prop1"].dim() == 1 and info["prop1"].size(0) == 12)
    assert(info["prop2"].dim() == 2 and info["prop2"].size(0) == 12 and info["prop2"].size(1) == 5)
    assert(info["prop3_x"].dim() == 1 and info["prop3_x"].size(0) == 12)
    assert(info["prop3_y"].dim() == 1 and info["prop3_y"].size(0) == 12)

def test_dict_of_torch():
    info = ExtraInfo(4, 'torch')
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

    info.parse(to='numpy')

    assert(len(info) == 3)
    assert(isinstance(info["prop1"], np.ndarray))
    assert(isinstance(info["prop2"], np.ndarray))
    assert(isinstance(info["prop3_x"], np.ndarray))
    assert(info["prop1"].ndim == 2 and info["prop1"].shape[0] == 2 and info["prop1"].shape[1] == 4)
    assert(info["prop2"].ndim == 3 and info["prop2"].shape[0] == 2 and info["prop2"].shape[1] == 4 and info["prop2"].shape[2] == 2)
    assert(info["prop3_x"].ndim == 2 and info["prop3_x"].shape[0] == 2 and info["prop3_x"].shape[1] == 4)

    info = info.flatten()

    assert(len(info) == 3)
    assert(isinstance(info["prop1"], np.ndarray))
    assert(isinstance(info["prop2"], np.ndarray))
    assert(isinstance(info["prop3_x"], np.ndarray))
    assert(info["prop1"].ndim == 1 and info["prop1"].shape[0] == 8)
    assert(info["prop2"].ndim == 2 and info["prop2"].shape[0] == 8 and info["prop2"].shape[1] == 2)
    assert(info["prop3_x"].ndim == 1 and info["prop3_x"].shape[0] == 8)

    assert np.array_equal(np.array([100, 110, 101, 111, 102, 112, 103, 113]), info["prop1"])
    prop2 = np.array([[200.0, 200.5], [210.0, 210.5], [201.0, 201.5], [211.0, 211.5], 
                      [202.0, 202.5], [212.0, 212.5], [203.0, 203.5], [213.0, 213.5]])
    assert np.array_equal(prop2, info["prop2"])
    assert np.array_equal(np.array([300, 310, 301, 311, 302, 312, 303, 313]), info["prop3_x"])

    info.parse()

    assert(len(info) == 3)
    assert(isinstance(info["prop1"], np.ndarray))
    assert(isinstance(info["prop2"], np.ndarray))
    assert(isinstance(info["prop3_x"], np.ndarray))
    assert(info["prop1"].ndim == 1 and info["prop1"].shape[0] == 8)
    assert(info["prop2"].ndim == 2 and info["prop2"].shape[0] == 8 and info["prop2"].shape[1] == 2)
    assert(info["prop3_x"].ndim == 1 and info["prop3_x"].shape[0] == 8)

def test_empty_dict_in_list():
    info = ExtraInfo(3, 'torch')

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
    assert(len(info) == 2)

    assert("prop1" in info)
    assert("prop2" in info)

    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop2"]))

    assert(info["prop1"].dim() == 1 and info["prop1"].size(0) == 3)
    assert(info["prop2"].dim() == 1 and info["prop2"].size(0) == 3)
    
    assert(info["prop1"][0] == 100 and info["prop2"][0] == 200)
    assert(torch.isnan(info["prop1"][1]) and torch.isnan(info["prop2"][1]))
    assert(info["prop1"][2] == 102 and info["prop2"][2] == 202)

def test_empty_dict():
    info = ExtraInfo(2, 'numpy')
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

    assert(len(info) == 1)
    assert("prop1" in info)
    assert(isinstance(info["prop1"], np.ndarray))
    assert(info["prop1"].ndim == 1 and info["prop1"].shape[0] == 6)

    assert info["prop1"][0] == 100
    assert np.isnan(info["prop1"][1])
    assert info["prop1"][2] == 120
    assert info["prop1"][3] == 101
    assert np.isnan(info["prop1"][4])
    assert info["prop1"][5] == 121

def test_changing_properties_dict():
    info = ExtraInfo(2, 'numpy')
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
    info.parse(to='torch')
    info = info.flatten()
    
    print(info)

    assert(len(info) == 3)

    assert("prop2" in info)
    assert("prop3" in info)
    assert("prop4" in info)

    assert(torch.is_tensor(info["prop2"]))
    assert(torch.is_tensor(info["prop3"]))
    assert(torch.is_tensor(info["prop4"]))

    assert(info["prop2"].dim() == 1 and info["prop2"].size(0) == 6)
    assert(info["prop3"].dim() == 1 and info["prop3"].size(0) == 6)
    assert(info["prop4"].dim() == 1 and info["prop4"].size(0) == 6)

    assert info["prop2"][0] == 200 and info["prop3"][0] == 300 and torch.isnan(info["prop4"][0])
    assert info["prop2"][1] == 210 and torch.isnan(info["prop3"][1]) and info["prop4"][1] == 410
    assert info["prop2"][2] == 220 and info["prop3"][2] == 320 and torch.isnan(info["prop4"][2])
    assert info["prop2"][3] == 201 and info["prop3"][3] == 301 and torch.isnan(info["prop4"][3])
    assert info["prop2"][4] == 211 and torch.isnan(info["prop3"][4]) and info["prop4"][4] == 411
    assert info["prop2"][5] == 221 and info["prop3"][5] == 321 and torch.isnan(info["prop4"][5])

def test_one_environment():
    info = ExtraInfo(1, 'torch')
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
    info.parse('torch')
    print(info)
    
    assert(len(info) == 3)

    assert("prop1" in info)
    assert("prop2" in info)
    assert("prop3" in info)

    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop2"]))
    assert(torch.is_tensor(info["prop3"]))

    assert(info["prop1"].dim() == 2 and info["prop1"].size(0) == 3 and info["prop2"].size(1) == 3)
    assert(info["prop2"].dim() == 3 and info["prop2"].size(0) == 3 and info["prop2"].size(1) == 3 and info["prop2"].size(2) == 2)
    assert(info["prop3"].dim() == 1 and info["prop3"].size(0) == 3)

def test_get_view_slice():
    info = ExtraInfo(3, 'torch')
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
    info.parse('torch')

    assert(len(info) == 2)

    assert("prop1" in info)
    assert("prop3" in info)

    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop3"]))

    assert(info["prop1"].dim() == 1 and info["prop1"].size(0) == 4)
    assert(info["prop3"].dim() == 2 and info["prop3"].size(0) == 4 and info["prop3"].size(1) == 2)

    assert(info["prop1"][0] == 100)
    assert(info["prop1"][1] == 110)
    assert(info["prop1"][2] == 101)
    assert(info["prop1"][3] == 111)

def test_get_view_array():
    info = ExtraInfo(3, 'torch')
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
    info.parse('torch')
    print(info)

    assert(len(info) == 2)

    assert("prop1" in info)
    assert("prop3" in info)

    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop3"]))

    assert(info["prop1"].dim() == 1 and info["prop1"].size(0) == 3)
    assert(info["prop3"].dim() == 2 and info["prop3"].size(0) == 3 and info["prop3"].size(1) == 2)

    assert(info["prop1"][0] == 110)
    assert(info["prop1"][1] == 101)
    assert(info["prop1"][2] == 112)

def test_add():
    info1 = ExtraInfo(10, 'numpy')
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

    info2 = ExtraInfo(10, 'torch')
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

    info1.parse('torch')
    info2.parse('numpy')

    info = info1 + info2

    assert(len(info) == 3)

    assert("prop1" in info)
    assert("prop2" in info)
    assert("prop3" in info)

    assert(torch.is_tensor(info["prop1"]))
    assert(torch.is_tensor(info["prop2"]))
    assert(torch.is_tensor(info["prop3"]))

    assert(info["prop1"].dim() == 2 and info["prop1"].size(0) == 4 and info["prop1"].size(1) == 10)
    assert(info["prop2"].dim() == 2 and info["prop2"].size(0) == 4 and info["prop2"].size(1) == 10)
    assert(info["prop3"].dim() == 2 and info["prop3"].size(0) == 4 and info["prop3"].size(1) == 10)

    for i in range(2):
        for j in range(10):
            assert(info["prop1"][i][j] == 100 + i*10 + j)
            assert(info["prop2"][i][j] == 200 + i*10 + j)
            assert(torch.isnan(info["prop3"][i][j]))
    
    for i in range(2):
        for j in range(10):
            assert(info["prop1"][2 + i][j] == 100 + i*10 + j)
            assert(torch.isnan(info["prop2"][2 + i][j]))
            assert(info["prop3"][2 + i][j] == 300 + i*10 + j)

def test_clear():
    info = ExtraInfo(10, 'numpy')
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
    assert(not info)

def test_flatten_with_mask():
    info = ExtraInfo(5, 'numpy')
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

    assert(len(info) == 2)

    assert("prop1" in info)
    assert("prop2" in info)

    assert(isinstance(info["prop1"], np.ndarray))
    assert(isinstance(info["prop2"], np.ndarray))

    assert(info["prop1"].ndim == 1 and info["prop1"].shape[0] == 4)
    assert(info["prop2"].ndim == 1 and info["prop2"].shape[0] == 4)

    assert np.array_equal(np.array([100, 110, 112, 104]), info["prop1"])
    assert np.array_equal(np.array([200, 210, 212, 204]), info["prop2"])

    #Test if mask is permantly applied
    info.parse()
    assert(len(info) == 2)

    assert("prop1" in info)
    assert("prop2" in info)

    assert(isinstance(info["prop1"], np.ndarray))
    assert(isinstance(info["prop2"], np.ndarray))

    assert(info["prop1"].ndim == 1 and info["prop1"].shape[0] == 4)
    assert(info["prop2"].ndim == 1 and info["prop2"].shape[0] == 4)

    assert np.array_equal(np.array([100, 110, 112, 104]), info["prop1"])
    assert np.array_equal(np.array([200, 210, 212, 204]), info["prop2"])
