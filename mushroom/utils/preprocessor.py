import numpy as np


class Preprocessor(object):
    """
    This is the interface class of the preprocessors.

    """
    def __call__(self, x):
        """
        Returns:
            The preprocessed input data array.

        """
        if isinstance(x, list):
            assert isinstance(x[0], np.ndarray)
            x[0] = self._compute(x[0])
        else:
            assert isinstance(x, np.ndarray)
            x = self._compute(x)

        return x


class Scaler(Preprocessor):
    """
    This class implements the function to scale the input data by a given
    coefficient.

    """
    def __init__(self, coeff):
        """
        Constructor.

        Args:
            coeff (float): the coefficient to use to scale input data.

        """
        self._coeff = coeff

    def _compute(self, x):
        """
        Args:
            x (np.array): input data

        Returns:
            The scaled input data array.

        """
        return x / self._coeff


class Binarizer(Preprocessor):
    """
    This class implements the function to binarize the values of an array
    according to a provided threshold value.

    """
    def __init__(self, threshold, geq=True):
        """
        Constructor.

        Args:
            threshold (float): the coefficient to use to scale input data.

        """
        self._threshold = threshold
        self._geq = geq

    def _compute(self, x):
        """
        Args:
            x (np.array): input data

        Returns:
            The binarized input data array.

        """
        if self._geq:
            return (x >= self._threshold).astype(np.float)
        else:
            return (x > self._threshold).astype(np.float)


class Filter(Preprocessor):
    """
    This class implements the function to filter the values of an array
    according to a provided array of indexes.

    """
    def __init__(self, idxs):
        """
        Constructor.

        Args:
            idxs (float): the array of idxs to use to filter input data.

        """
        self._idxs = idxs

    def _compute(self, x):
        """
        Args:
            x (np.array): input data

        Returns:
            The filter input data array.

        """
        return x[self._idxs]
