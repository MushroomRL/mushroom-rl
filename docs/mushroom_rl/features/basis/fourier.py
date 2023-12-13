import numpy as np


class FourierBasis:
    r"""
    Class implementing Fourier basis functions. The value of the feature
    is computed using the formula:

    .. math::
        \sum \cos{\pi(X - m)/\Delta c}

    where :math:`X` is the input, m is the vector of the minumum input values (for each dimensions) , :math:`\Delta` is
    the vector of differences between maximum and minumun values for the variables.

    """
    def __init__(self, low, delta, c, dimensions=None):
        """
        Constructor.

        Args:
            low (np.ndarray): vector of minimum values of the input variables;
            delta (np.ndarray): vector of the maximum difference between two values of the input variables, i.e.
                delta = high - low;
            c (np.ndarray): vector of weights for the state variables;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature.

        """
        self._low = low
        self._delta = delta
        self._dim = dimensions
        self._c = c

    def __call__(self, x):
        if self._dim is not None:
            x = x[self._dim]

        s = (x - self._low) / self._delta

        return np.cos(np.pi*s.dot(self._c))

    def __str__(self):
        return str(self._c)

    @staticmethod
    def generate(low, high, n, dimensions=None):
        """
        Factory method to build a set of fourier basis.

        Args:
            low (np.ndarray): vector of minimum values of the input variables;
            high (np.ndarray): vector of maximum values of the input variables;
            n (int): number of harmonics to consider for each state variable
            dimensions (list, None): list of the dimensions of the input to be
                considered by the features.

        Returns:
            The list of the generated fourier basis functions.

        """
        if dimensions is not None:
            assert(len(low) == len(dimensions))

        input_size = len(low)
        delta = high - low
        n_basis = (n + 1)**input_size
        basis_list = list()

        for index in range(n_basis):
            c = np.zeros(input_size)
            value = index

            for i in range(input_size):
                c[i] = value % (n + 1)
                value = value // (n + 1)

            basis_list.append(FourierBasis(low, delta, c, dimensions))

        return basis_list
