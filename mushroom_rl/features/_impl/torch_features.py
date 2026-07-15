import torch
import torch.nn as nn

from mushroom_rl.features.features import Features


class TorchFeatures(Features):
    class TorchFeatureModule(nn.Module):
        """
        Differentiable torch module concatenating the outputs of a list of basis tensors. It is returned by
        ``TorchFeatures.to_torch_module`` for embedding the features inside a network with gradient.

        """
        def __init__(self, tensor_list):
            super().__init__()

            self._phi = nn.ModuleList(tensor_list)

        def forward(self, x):
            return torch.cat([phi(x) for phi in self._phi], dim=1)

        @property
        def size(self):
            return sum(phi.size for phi in self._phi)

    def __init__(self, feature_list=None, backend='numpy'):
        self._phi = self.TorchFeatureModule(self._as_list(feature_list))

        super().__init__(backend, internal_backend='torch')

        self._add_save_attr(_phi='torch')

    def to_torch_module(self):
        return self._phi

    @property
    def size(self):
        return self._phi.size

    def _compute(self, x):
        with torch.no_grad():
            y = self._phi(torch.atleast_2d(x))

        if x.dim() == 1:
            return y.squeeze(0)

        return y
