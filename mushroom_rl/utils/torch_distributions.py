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
        self._delta = .5 * (high - low)
        self._central = .5 * (high + low)
        self._eps = eps

        base = Independent(Normal(loc, scale), 1)
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=self._central, scale=self._delta)]

        super().__init__(base, transforms, validate_args=validate_args)

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
