import torch
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform, AffineTransform


class CategoricalWrapper(torch.distributions.Categorical):
    """
    Wrapper for the Torch Categorical distribution.

    Needed to convert a vector of mushroom discrete action in an input with the proper shape of the original
    distribution implemented in torch

    """
    def __init__(self, logits):
        super().__init__(logits=logits)

    def log_prob(self, value):
        return super().log_prob(value.squeeze())


class SquashedGaussian(TransformedDistribution):
    """
    Diagonal Gaussian distribution squashed by a tanh and remapped to a bounded action range.

    The distribution lives in the action space ``[low, high]``: a sample is drawn from a diagonal Gaussian, squashed
    by a tanh into ``(-1, 1)`` and finally affinely remapped to ``[low, high]``. The proper change-of-variables is
    handled by the underlying transforms, so ``log_prob`` is a correct density in the action space.

    """
    def __init__(self, loc, scale, low, high, eps=1e-6, validate_args=False):
        """
        Constructor.

        Args:
            loc (torch.Tensor): mean of the underlying Gaussian;
            scale (torch.Tensor): standard deviation of the underlying Gaussian;
            low (torch.Tensor): minimum value for each action component;
            high (torch.Tensor): maximum value for each action component;
            eps (float, 1e-6): small constant used to keep the tanh inverse and its log finite.

        """
        self._low = low
        self._high = high
        self._delta = .5 * (high - low)
        self._central = .5 * (high + low)
        self._eps = eps

        base = Independent(Normal(loc, scale), 1)
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=self._central, scale=self._delta)]

        super().__init__(base, transforms, validate_args=validate_args)

    @property
    def median(self):
        """
        The median of the squashed distribution, i.e. the base Gaussian mean pushed through the tanh and affine
        transforms.

        """
        return torch.tanh(self.base_dist.mean) * self._delta + self._central

    def log_prob(self, value):
        a_squashed = torch.clamp((value - self._central) / self._delta, -1. + self._eps, 1. - self._eps)
        value = a_squashed * self._delta + self._central

        return super().log_prob(value)

    def rsample_and_log_prob(self):
        """
        Sample an action using the reparametrization trick and compute its log probability directly, without
        inverting the tanh, to avoid the precision loss caused by the inverse near the boundaries.

        Returns:
            The sampled action and its log probability.

        """
        a_raw = self.base_dist.rsample()
        a_tanh = torch.tanh(a_raw)
        a = a_tanh * self._delta + self._central

        log_prob = self.base_dist.log_prob(a_raw)
        log_prob = log_prob - torch.log(1. - a_tanh.pow(2) + self._eps).sum(dim=-1)
        log_prob = log_prob - torch.log(self._delta).sum()

        return a, log_prob

    @property
    def low(self):
        """
        Returns:
            The minimum value of each action component.

        """
        return self._low

    @property
    def high(self):
        """
        Returns:
            The maximum value of each action component.

        """
        return self._high

    @property
    def eps(self):
        """
        Returns:
            The constant keeping the tanh inverse and its log finite.

        """
        return self._eps


class DistHelperWrapper:
    """
    Wrapper of a batched torch distribution providing tensor operations on its first batch dimension.

    Currently, the supported distributions are ``torch.distributions.MultivariateNormal``, :class:`CategoricalWrapper`
    and :class:`SquashedGaussian`.

    """
    def __init__(self, distribution):
        """
        Constructor.

        Args:
            distribution (torch.distributions.Distribution): the distribution to wrap, with the samples along its
                first batch dimension.

        Raises:
            NotImplementedError: if the type of the distribution is not supported.

        """
        self._distribution = distribution
        self._parameters = self._split()

    def __len__(self):
        """
        Returns:
            The number of samples of the wrapped distribution.

        """
        return len(self._parameters[0])

    def __getitem__(self, index):
        """
        Args:
            index: the samples to select, indexing the first batch dimension like a tensor index.

        Returns:
            A distribution of the same type as the wrapped one, restricted to the selected samples.

        """
        return self._merge([parameter[index] for parameter in self._parameters])

    @property
    def distribution(self):
        """
        Returns:
            The wrapped distribution.

        """
        return self._distribution

    def _split(self):
        distribution = self._distribution

        if type(distribution) is torch.distributions.MultivariateNormal:
            return distribution.loc, distribution.scale_tril
        elif type(distribution) is CategoricalWrapper:
            return distribution.logits,
        elif type(distribution) is SquashedGaussian:
            normal = distribution.base_dist.base_dist
            return normal.loc, normal.scale

        raise NotImplementedError(f'The {type(distribution).__name__} distribution is not supported')

    def _merge(self, parameters):
        distribution = self._distribution

        if type(distribution) is torch.distributions.MultivariateNormal:
            loc, scale_tril = parameters
            return torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril, validate_args=False)
        elif type(distribution) is CategoricalWrapper:
            logits, = parameters
            return CategoricalWrapper(logits)
        else:
            loc, scale = parameters
            return SquashedGaussian(loc, scale, distribution.low, distribution.high, eps=distribution.eps)
