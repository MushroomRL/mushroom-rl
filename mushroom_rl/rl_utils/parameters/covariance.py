from mushroom_rl.rl_utils.parameters.parameter import Parameter


class CovarianceParameter(Parameter):
    """
    Interface for a covariance matrix. Every value is returned in the array backend and on the device of the parameter.
    :meth:`make` builds the appropriate implementation from a covariance matrix or a standard deviation.

    """
    def __init__(self, value, backend='numpy', device=None):
        """
        Constructor.

        Args:
            value ([Array, None]): fixed covariance matrix, if any;
            backend (str, 'numpy'): array backend of the parameter, either ``'numpy'`` or ``'torch'``;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        assert backend in ['numpy', 'torch'], f'Unsupported backend {backend}.'

        super().__init__(value, backend=backend, device=device)

    def get_value(self, *args, **kwargs):
        """
        Returns:
            The current covariance matrix.

        """
        raise NotImplementedError

    def set_backend(self, backend, device=None):
        """
        Set the array backend and the device of the parameter.

        Args:
            backend (str): name of the array backend, either ``'numpy'`` or ``'torch'``;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        assert backend in ['numpy', 'torch'], f'Unsupported backend {backend}.'

        super().set_backend(backend, device)

    def get_cholesky(self):
        """
        Returns:
            The lower triangular Cholesky factor of the current covariance matrix.

        """
        raise NotImplementedError

    def get_inverse(self):
        """
        Returns:
            The inverse of the current covariance matrix.

        """
        raise NotImplementedError

    def get_std(self):
        """
        Returns:
            The current standard deviation of each dimension, i.e. the square root of the diagonal of the covariance
            matrix.

        """
        raise NotImplementedError

    @staticmethod
    def make(x, backend='numpy', device=None, correlation=None):
        """
        Args:
            x ([Array, Parameter, CovarianceParameter]): a covariance matrix, a vector of standard deviations, or a
                :class:`Parameter` returning the vector of standard deviations;
            backend (str, 'numpy'): array backend of the parameter, either ``'numpy'`` or ``'torch'``;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used;
            correlation (Array, None): correlation matrix, of size n x n; if given, a
                :class:`CorrelatedCovarianceParameter` is built from the standard deviations ``x``.

        Returns:
            ``x`` in the given backend if it is already a :class:`CovarianceParameter`, otherwise the appropriate
            :class:`CovarianceParameter` built from it.

        """
        if isinstance(x, CovarianceParameter):
            assert correlation is None, 'A correlation cannot be given with an existing CovarianceParameter.'
            x.set_backend(backend, device)
            return x

        is_matrix = not isinstance(x, Parameter) and x.ndim == 2
        if correlation is not None:
            assert not is_matrix, 'A correlation can be given only together with standard deviations.'
            return CorrelatedCovarianceParameter(x, correlation, backend=backend, device=device)
        elif is_matrix:
            return FixedCovarianceParameter(x, backend=backend, device=device)
        else:
            return DiagonalCovarianceParameter(x, backend=backend, device=device)

    def _log_value(self, *args, **kwargs):
        return self.get_std()


class FixedCovarianceParameter(CovarianceParameter):
    """
    Class for defining a fixed covariance matrix.

    """
    def __init__(self, value, backend='numpy', device=None):
        """
        Constructor.

        Args:
            value (Array): covariance matrix, of size n x n;
            backend (str, 'numpy'): array backend of the parameter, either ``'numpy'`` or ``'torch'``;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        super().__init__(value, backend=backend, device=device)

        self._cholesky = None
        self._inverse = None

        self._add_save_attr(
            _cholesky='none',
            _inverse='none'
        )

    def get_value(self, *args, **kwargs):
        return self._initial_value

    def set_backend(self, backend, device=None):
        super().set_backend(backend, device)

        self._cholesky = None
        self._inverse = None

    def get_cholesky(self):
        if self._cholesky is None:
            self._cholesky = self._backend.cholesky(self._initial_value)

        return self._cholesky

    def get_inverse(self):
        if self._inverse is None:
            self._inverse = self._backend.inv(self._initial_value)

        return self._inverse

    def get_std(self):
        return self._backend.sqrt(self._backend.diagonal(self._initial_value))


class DiagonalCovarianceParameter(CovarianceParameter):
    """
    Class for defining a diagonal covariance matrix from the standard deviation of each dimension, optionally
    scheduled by a :class:`Parameter`.

    Updating the parameter advances the standard deviation schedule.

    """
    def __init__(self, std, backend='numpy', device=None):
        """
        Constructor.

        Args:
            std ([Array, Parameter]): standard deviation of each of the n dimensions, as a vector or as a
                :class:`Parameter` returning it;
            backend (str, 'numpy'): array backend of the parameter, either ``'numpy'`` or ``'torch'``;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        super().__init__(None, backend=backend, device=device)

        self._std = Parameter.make(std, backend=backend, device=device)

        self._add_save_attr(_std='mushroom')

    def get_value(self, *args, **kwargs):
        return self._backend.diag(self.get_std() ** 2)

    def update(self, *args, **kwargs):
        self._std.update()

        super().update()

    def set_backend(self, backend, device=None):
        super().set_backend(backend, device)

        self._std.set_backend(backend, device)

    def get_cholesky(self):
        return self._backend.diag(self.get_std())

    def get_inverse(self):
        return self._backend.diag(1. / self.get_std() ** 2)

    def get_std(self):
        return self._std.get_value()


class CorrelatedCovarianceParameter(DiagonalCovarianceParameter):
    """
    Class for defining a covariance matrix from the standard deviation of each dimension, optionally scheduled by a
    :class:`Parameter`, and a fixed correlation matrix.

    Updating the parameter advances the standard deviation schedule.

    """
    def __init__(self, std, correlation, backend='numpy', device=None):
        """
        Constructor.

        Args:
            std ([Array, Parameter]): standard deviation of each of the n dimensions, as a vector or as a
                :class:`Parameter` returning it;
            correlation (Array): correlation matrix, of size n x n;
            backend (str, 'numpy'): array backend of the parameter, either ``'numpy'`` or ``'torch'``;
            device (str, None): device of the parameter, only for the torch backend. If None, the default device
                is used.

        """
        super().__init__(std, backend=backend, device=device)

        self._correlation = self._to_backend(correlation)
        self._correlation_cholesky = None
        self._correlation_inverse = None

        self._add_save_attr(
            _correlation=self._value_serialization_method(self._correlation),
            _correlation_cholesky='none',
            _correlation_inverse='none'
        )

    def get_value(self, *args, **kwargs):
        std = self.get_std()

        return std[:, None] * self._correlation * std[None, :]

    def set_backend(self, backend, device=None):
        super().set_backend(backend, device)

        self._correlation = self._to_backend(self._correlation)
        self._correlation_cholesky = None
        self._correlation_inverse = None

        self._add_save_attr(_correlation=self._value_serialization_method(self._correlation))

    def get_cholesky(self):
        if self._correlation_cholesky is None:
            self._correlation_cholesky = self._backend.cholesky(self._correlation)

        return self.get_std()[:, None] * self._correlation_cholesky

    def get_inverse(self):
        if self._correlation_inverse is None:
            self._correlation_inverse = self._backend.inv(self._correlation)

        std = self.get_std()

        return self._correlation_inverse / (std[:, None] * std[None, :])
