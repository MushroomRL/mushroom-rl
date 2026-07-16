import numpy as np

from .basis_function import BasisFunction


class FourierBasis(BasisFunction):
    r"""
    Class implementing Fourier basis functions. The value of the feature
    is computed using the formula:

    .. math::
        \cos{\left(\pi \sum_i \dfrac{(X_i - m_i)}{\Delta_i} c_i\right)}

    where :math:`X` is the input, :math:`m` is the vector of the minimum input values (for each dimension),
    :math:`\Delta` is the vector of differences between maximum and minimum values for the variables and
    :math:`c` is the vector of weights.

    """
    def __init__(self, low, delta, c, dimensions=None):
        """
        Constructor.

        Args:
            low (Array): vector of minimum values of the input variables;
            delta (Array): vector of the maximum difference between two values of the input variables, i.e.
                delta = high - low;
            c (Array): vector of weights for the state variables;
            dimensions (list, None): list of the dimensions of the input to be considered by the feature.

        """
        self._low, self._delta, self._c = self._to_numpy(low, delta, c)

        super().__init__(dimensions)

        self._add_save_attr(_low='numpy', _delta='numpy', _c='numpy')

    def __repr__(self):
        return f'FourierBasis(c={np.array2string(self._c)})'

    @staticmethod
    def generate(low, high, n, dimensions=None):
        """
        Factory method to build a set of fourier basis.

        Args:
            low (Array): vector of minimum values of each variable of the whole input;
            high (Array): vector of maximum values of each variable of the whole input;
            n (int): number of harmonics to consider for each selected input variable;
            dimensions (list, None): list of the dimensions of the input to be considered by the features. If
                None, every dimension of the input is used.

        Returns:
            The list of the generated fourier basis functions.

        """
        low, high = BasisFunction._to_numpy(low, high)

        assert len(low) == len(high)

        if dimensions is not None:
            low, high = low[dimensions], high[dimensions]

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

    def _compute(self, x):
        s = (x - self._low) / self._delta

        return np.cos(np.pi*s.dot(self._c))
