import numpy as np

from mushroom_rl.utils.sum_tree import SumTree


def test_sum_tree_update_priorities():
    n = 5
    priorities = np.array([1.0, 2.0, 1.0, 1.5, 0.5])

    tree = SumTree(max_size=20)
    tree_idxs = np.arange(n) + tree._max_size - 1
    tree.update(tree_idxs, priorities)

    assert np.isclose(tree.total_p, 6.0)
    assert np.isclose(tree.max_p, 2.0)


def test_sum_tree_update_propagates():
    n = 5

    tree = SumTree(max_size=20)
    tree_idxs = np.arange(n) + tree._max_size - 1
    tree.update(tree_idxs, np.ones(n))

    first_leaf = tree._max_size - 1
    tree.update([first_leaf], [5.0])

    assert np.isclose(tree.total_p, 9.0)
    assert np.isclose(tree.max_p, 5.0)
    assert np.isclose(tree._tree[first_leaf], 5.0)


def test_sum_tree_get_retrieves_correct_priority():
    n = 5
    priorities = np.array([1.0, 2.0, 1.0, 1.5, 0.5])

    tree = SumTree(max_size=20)
    tree_idxs = np.arange(n) + tree._max_size - 1
    tree.update(tree_idxs, priorities)

    idx, p = tree.get(0.5)
    assert idx - (tree._max_size - 1) == 0
    assert np.isclose(p, 1.0)

    idx2, p2 = tree.get(3.0)
    assert idx2 - (tree._max_size - 1) == 1
    assert np.isclose(p2, 2.0)


def test_sum_tree_mask_never_sampled():
    n = 8
    leaf0 = n - 1

    tree = SumTree(max_size=n)
    tree.update(np.arange(n) + leaf0, np.ones(n))
    masked = np.array([3, 4]) + leaf0
    tree.mask(masked)

    assert np.isclose(tree.total_p, 6.0)

    np.random.seed(0)
    segment = tree.total_p / 5000
    drawn = {tree.get(np.random.uniform(k * segment, (k + 1) * segment))[0] for k in range(5000)}

    assert drawn.isdisjoint(set(masked.tolist()))


def test_sum_tree_update_unmasks():
    n = 8
    leaf0 = n - 1

    tree = SumTree(max_size=n)
    tree.update(np.arange(n) + leaf0, np.ones(n))
    tree.mask(np.array([3 + leaf0]))

    assert tree._masked[3 + leaf0]
    assert np.isclose(tree.total_p, 7.0)

    tree.update(np.array([3 + leaf0]), np.array([2.0]))

    assert not tree._masked[3 + leaf0]
    assert np.isclose(tree.total_p, 9.0)
