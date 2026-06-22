import numpy as np
import torch


def minibatch_number(size, batch_size):
    """
    Function to retrieve the number of batches, given a batch sizes.

    Args:
        size (int): size of the dataset;
        batch_size (int): size of the batches.

    Returns:
        The number of minibatches in the dataset.

    """
    return int(np.ceil(size / batch_size))


def minibatch_generator(batch_size, *dataset):
    """
    Generator that creates a minibatch from the full dataset.

    Args:
        batch_size (int): the maximum size of each minibatch;
        dataset: the dataset to be splitted.

    Returns:
        The current minibatch.

    """
    size = len(dataset[0])
    num_batches = minibatch_number(size, batch_size)
    indexes = np.arange(0, size, 1)
    np.random.shuffle(indexes)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]

    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset)):
            batch.append(dataset[i][indexes[batch_start:batch_end]])
        yield batch


def ensemble_minibatch_generator(batch_size, n_models, *dataset):
    """
    Generator that creates independently-shuffled minibatches for ensemble training.
    Each model gets its own shuffle of the dataset; batches are
    then stacked so that all models are processed together in a single ``vmap`` call.

    Args:
        batch_size (int): the maximum size of each minibatch;
        n_models (int): number of ensemble models;
        dataset: the dataset to be split.

    Returns:
        For each batch index, a list of stacked arrays with shape (n_models, batch_size, ...).

    """
    size = len(dataset[0])
    num_batches = minibatch_number(size, batch_size)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(num_batches)]

    all_indexes = [torch.randperm(size) for _ in range(n_models)]

    for batch_start, batch_end in batches:
        yield [
            torch.stack([dataset[j][all_indexes[m][batch_start:batch_end]] for m in range(n_models)])
            for j in range(len(dataset))
        ]
