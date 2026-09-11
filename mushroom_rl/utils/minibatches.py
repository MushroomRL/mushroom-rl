import math
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
    return math.ceil(size / batch_size)


def minibatch_generator(batch_size, *dataset_vectors):
    """
    Generator that creates a minibatch from the full dataset.

    Args:
        batch_size (int): the maximum size of each minibatch;
        dataset_vectors: the torch tensors to be split.

    Returns:
        The current minibatch.

    """
    size = len(dataset_vectors[0])
    num_batches = minibatch_number(size, batch_size)
    indexes = torch.randperm(size)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]

    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset_vectors)):
            batch.append(dataset_vectors[i][indexes[batch_start:batch_end]])
        yield batch


def ensemble_minibatch_generator(batch_size, n_models, *dataset_vectors):
    """
    Generator that creates independently-shuffled minibatches for ensemble training.
    Each model gets its own shuffle of the dataset; batches are
    then stacked so that all models are processed together in a single ``vmap`` call.

    Args:
        batch_size (int): the maximum size of each minibatch;
        n_models (int): number of ensemble models;
        dataset_vectors: the torch tensors to be split.

    Returns:
        For each batch index, a list of stacked arrays with shape (n_models, batch_size, ...).

    """
    size = len(dataset_vectors[0])
    num_batches = minibatch_number(size, batch_size)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(num_batches)]

    all_indexes = [torch.randperm(size) for _ in range(n_models)]

    for batch_start, batch_end in batches:
        yield [
            torch.stack([dataset_vectors[j][all_indexes[m][batch_start:batch_end]] for m in range(n_models)])
            for j in range(len(dataset_vectors))
        ]
