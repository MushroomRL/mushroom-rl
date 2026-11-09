import torch.nn as nn


class MushroomNetwork(nn.Module):
    """
    Base class for the MushroomRL networks.

    """
    def reset_parameters(self):
        """
        Draw new initial values for all the parameters of the network, from the same distribution used at
        construction. Every submodule holding parameters must define a ``reset_parameters`` method, or be a
        ``nn.ModuleList``, ``nn.Sequential`` or ``nn.ModuleDict`` whose elements satisfy the same condition.

        """
        for child in self.children():
            self._reset_module(child)
        self._init_weights()

    def _init_weights(self):
        """
        Apply the initialization of the network on top of the default initialization of its submodules.
        Subclasses call it at the end of their constructor; ``reset_parameters`` calls it after resetting the
        submodules.

        """
        pass

    @staticmethod
    def _reset_module(module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        elif isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            for child in module.children():
                MushroomNetwork._reset_module(child)
        elif next(module.parameters(), None) is not None:
            raise TypeError(f'Cannot reset the parameters of a {type(module).__name__}: it defines no '
                            f'reset_parameters method.')
