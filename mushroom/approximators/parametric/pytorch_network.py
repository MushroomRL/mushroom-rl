import torch
import numpy as np
from tqdm import trange, tqdm

from mushroom.utils.minibatches import minibatch_generator


class PyTorchApproximator:
    """
    Class to interface a pytorch model to the mushroom Regressor interface.
    This class implements all is needed to use a generic pytorch model and train
    it using a specified optimizer and objective function.
    This class supports also minibatches.

    """
    def __init__(self, input_shape, output_shape, network, optimizer, loss,
                 n_epochs=1, batch_size=0, device=None, quiet=True, **params):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input of the network;
            output_shape (tuple): shape of the output of the network;
            network (torch.nn.Module): the network class to use;
            optimizer (dict): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the
                fit method;
            n_epochs (int, 1): the number of epochs to run during the fit;
            batch_size (int, 0): the size of each minibatch. If 0, the whole
                dataset is fed to the optimizer at each epoch;
            device (int, None): if None, runs the network on the CPU;
                otherwise the integer provided specifies the GPU device to use;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            params (dict): dictionary of parameters needed to construct the
                network.

        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._device = device
        self._quiet = quiet

        params['device'] = self._device
        self._network = network(input_shape, output_shape, **params)
        if self._device is not None:
            self._network.cuda(self._device)

        self._optimizer = optimizer['class'](self._network.parameters(),
                                             **optimizer['params'])
        self._loss = loss

    def predict(self, *args, **kwargs):
        if self._device is None:
            torch_args = [torch.from_numpy(x) for x in args]
            val = self._network.forward(*torch_args, **kwargs).detach().numpy()
        else:
            torch_args = [torch.from_numpy(x).cuda(self._device) for x in args]
            val = self._network.forward(*torch_args,
                                        **kwargs).detach().cpu().numpy()

        return val

    def fit(self, *args, **kwargs):
        for t in trange(self._n_epochs, disable=self._quiet):
            if self._batch_size > 0:
                batches = minibatch_generator(self._batch_size, *args)
            else:
                batches = [args]

            loss_current = list()
            for batch in tqdm(batches, disable=self._quiet):
                if self._device is None:
                    torch_args = [torch.from_numpy(x) for x in batch]
                else:
                    torch_args = [torch.from_numpy(x).cuda(self._device) for x
                                  in args]

                x = torch_args[:-1]
                y_hat = self._network(*x, **kwargs)
                y = torch.tensor(torch_args[-1], dtype=y_hat.dtype)
                if self._device is not None:
                    y = y.cuda(self._device)

                loss = self._loss(y_hat, y)
                loss_current.append(loss.item())

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            if not self._quiet:
                tqdm.write('loss at epoch ' + str(t) + ' ' +
                           str(np.mean(loss_current)))

    def set_weights(self, weights):
        self._network.load_state_dict(weights)

    def get_weights(self):
        return self._network.state_dict()
