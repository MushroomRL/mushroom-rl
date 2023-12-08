import numpy as np


class PolynomialBasis:
    r"""
    Class implementing polynomial basis functions. The value of the feature
    is computed using the formula:
    
    .. math::
        \prod X_i^{d_i}

    where :math:`X~ is the input and :math:`d` is the vector of the exponents of the polynomial.

    """
    def __init__(self, dimensions=None, degrees=None, low=None, high=None):
        """
        Constructor. If both parameters are None, the constant feature is built.

        Args:
            dimensions (list, None): list of the dimensions of the input to be considered by the feature;
            degrees (list, None): list of the degrees of each dimension to be considered by the feature.
                It must match the number of elements of ``dimensions``;
            low (np.ndarray, None): array specifying the lower bound of the action space, used to normalize the
                state between -1 and 1;
            high (np.ndarray, None): array specifying the lower bound of the action space, used to normalize the
                state between -1 and 1;

        """
        assert (dimensions is None and degrees is None) or (
                len(dimensions) == len(degrees))

        assert (low is None and high is None) or (low is not None and high is not None)

        self._dim = dimensions
        self._deg = degrees

        if low is not None:
            self._mean = (low + high) / 2
            self._delta = (high - low) / 2
        else:
            self._mean = None

    def __call__(self, x):

        if self._dim is None:
            return 1

        out = 1
        x_n = self._normalize(x)
        for i, d in zip(self._dim, self._deg):
            out *= x_n[i]**d

        return out

    def _normalize(self, x):
        if self._mean is not None:
            return (x - self._mean) / self._delta
        return x

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
    def generate(max_degree, input_size, low=None, high=None):
        """
        Factory method to build a polynomial of order ``max_degree`` based on
        the first ``input_size`` dimensions of the input.

        Args:
            max_degree (int): maximum degree of the polynomial;
            input_size (int): size of the input;
            low (np.ndarray, None): array specifying the lower bound of the action space, used to normalize the
                state between -1 and 1;
            high (np.ndarray, None): array specifying the lower bound of the action space, used to normalize the
                state between -1 and 1;

        Returns:
            The list of the generated polynomial basis functions.

        """
        assert (max_degree >= 0)
        assert (input_size > 0)

        basis_list = [PolynomialBasis()]

        for e in PolynomialBasis._compute_exponents(max_degree, input_size):
            dims = np.reshape(np.argwhere(e != 0), -1)
            degs = e[e != 0]

            basis_list.append(PolynomialBasis(dims, degs, low, high))

        return basis_list
