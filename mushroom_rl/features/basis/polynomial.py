import numpy as np


class PolynomialBasis:
    r"""
    Class implementing polynomial basis functions. The value of the feature
    is computed using the formula:
    
    .. math::
        \prod X_i^{d_i}

    where X is the input and d is the vector of the exponents of the polynomial.

    """
    def __init__(self, dimensions=None, degrees=None):
        """
        Constructor. If both parameters are None, the constant feature is built.

        Args:
            dimensions (list, None): list of the dimensions of the input to be
                considered by the feature;
            degrees (list, None): list of the degrees of each dimension to be
                considered by the feature. It must match the number of elements
                of ``dimensions``.

        """
        self._dim = dimensions
        self._deg = degrees

        assert (self._dim is None and self._deg is None) or (
            len(self._dim) == len(self._deg))

    def __call__(self, x):

        if self._dim is None:
            return 1

        out = 1
        for i, d in zip(self._dim, self._deg):
            out *= x[i]**d

        return out

    def __str__(self):
        if self._deg is None:
            return '1'

        name = ''
        for i, d in zip(self._dim, self._deg):
            name += 'x[' + str(i) + ']'
            if d > 1:
                name += '^' + str(d)
        return name

    @staticmethod
    def _compute_exponents(order, n_variables):
        """
        Find the exponents of a multivariate polynomial expression of order
        ``order`` and ``n_variables`` number of variables.

        Args:
            order (int): the maximum order of the polynomial;
            n_variables (int): the number of elements of the input vector.

        Yields:
            The current exponent of the polynomial.

        """
        pattern = np.zeros(n_variables, dtype=int)
        for current_sum in range(1, order + 1):
            pattern[0] = current_sum
            yield pattern
            while pattern[-1] < current_sum:
                for i in range(2, n_variables + 1):
                    if 0 < pattern[n_variables - i]:
                        pattern[n_variables - i] -= 1
                        if 2 < i:
                            pattern[n_variables - i + 1] = 1 + pattern[-1]
                            pattern[-1] = 0
                        else:
                            pattern[-1] += 1
                        break
                yield pattern
            pattern[-1] = 0

    @staticmethod
    def generate(max_degree, input_size):
        """
        Factory method to build a polynomial of order ``max_degree`` based on
        the first ``input_size`` dimensions of the input.

        Args:
            max_degree (int): maximum degree of the polynomial;
            input_size (int): size of the input.

        Returns:
            The list of the generated polynomial basis functions.

        """
        assert (max_degree >= 0)
        assert (input_size > 0)

        basis_list = [PolynomialBasis()]

        for e in PolynomialBasis._compute_exponents(max_degree, input_size):
            dims = np.reshape(np.argwhere(e != 0), -1)
            degs = e[e != 0]

            basis_list.append(PolynomialBasis(dims, degs))

        return basis_list
