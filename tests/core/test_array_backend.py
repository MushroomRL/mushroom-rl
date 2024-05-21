import numpy.random
import torch
import numpy as np
from mushroom_rl.core.array_backend import NumpyBackend, TorchBackend


def sequence_generator():
    list_n_steps = np.random.randint(2, 10, 100)
    list_n_envs = np.random.randint(2, 20, 100)
    list_n_dim = np.random.randint(1, 10, 100)

    for n_steps, n_envs, n_dim in zip(list_n_steps, list_n_envs, list_n_dim):
        lengths = np.random.randint(1, n_steps, size=(n_envs,))

        array = list()

        for d in range(n_dim):
            offset_dim = 10 * d

            array_dim = list()
            for e in range(n_envs):
                offset_env = 100 * e
                array_env_dim = offset_dim + offset_env + np.arange(0, n_steps)

                array_dim.append(array_env_dim)

            array_dim = np.stack(array_dim).T

            array.append(array_dim)

        array = np.stack(array, axis=-1).squeeze()

        mask = (np.arange(len(array))[:, None] < lengths[None, :])

        yield array, mask, lengths


def test_packed_2d_sequence_numpy():
    print('testing 2d sequence')
    array = np.arange(0, 100).reshape(20, 5, order='F')
    desired = np.concatenate([np.arange(0, 60), np.arange(60, 70), np.arange(80, 90)])
    mask = np.ones(100, dtype=bool).reshape(20, 5, order='F')

    mask[10:, 3:] = False

    print(mask)

    packed = NumpyBackend.pack_padded_sequence(array, mask)

    print('array')
    print(array)

    print('packed')
    print(packed)
    print('desired')
    print(desired)

    assert (packed == desired).all()


def test_pack_sequence_numpy():
    numpy.random.seed(42)

    for array, mask, lengths in sequence_generator():
        print('################################## Numpy')
        print('original')
        print(array)

        print('lengths')
        print(lengths)

        packed_array = NumpyBackend.pack_padded_sequence(array, mask)
        print('packed')
        print(packed_array)

        desired_array = np.concatenate([array[:l, i] for i, l in enumerate(lengths)])
        print('desired')
        print(desired_array)

        assert np.array_equal(desired_array, packed_array)


def test_packed_2d_sequence_torch():
    print('testing 2d sequence')
    array = torch.arange(0, 100).reshape(5, 20).T
    desired = torch.concatenate([torch.arange(0, 60), torch.arange(60, 70), torch.arange(80, 90)])
    mask = torch.ones(100, dtype=torch.bool).reshape(20, 5)

    mask[10:, 3:] = False
    print('mask')
    print(mask)

    packed = TorchBackend.pack_padded_sequence(array, mask)

    print('array')
    print(array)

    print('packed')
    print(packed)
    print('desired')
    print(desired)

    assert (packed == desired).all()


def test_pack_sequence_torch():
    numpy.random.seed(42)

    for array, mask, lengths in sequence_generator():
        torch_array = torch.as_tensor(array)
        mask = torch.as_tensor(mask)

        print('original')
        print(torch_array)

        print('lengths')
        print(lengths)

        packed_array = TorchBackend.pack_padded_sequence(torch_array, mask)
        print('packed')
        print(packed_array)

        desired_array = torch.concatenate([torch_array[:l, i] for i, l in enumerate(lengths)])
        print('desired')
        print(desired_array)

        assert torch.equal(packed_array, desired_array)
