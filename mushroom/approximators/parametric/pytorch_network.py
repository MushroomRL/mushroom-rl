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
    def __init__(self, input_shape, output_shape, network, optimizer=None,
                 loss=None, n_epochs=1, batch_size=0, use_cuda=False,
                 dropout=False, quiet=True, n_fit_targets=1, **params):
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
            use_cuda (bool, False): if True, runs the network on the GPU;
                otherwise the integer provided specifies the GPU device to use;
            dropout (bool, False): if True, dropout is applied only during
                train;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            params (dict): dictionary of parameters needed to construct the
                network.

        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._quiet = quiet
        self._n_fit_targets = n_fit_targets

        self._network = network(input_shape, output_shape, use_cuda=use_cuda,
                                dropout=dropout, **params)
        if self._use_cuda:
            self._network.cuda()
        if self._dropout:
            self._network.eval()

        if optimizer is not None:
            self._optimizer = optimizer['class'](self._network.parameters(),
                                                 **optimizer['params'])
        self._loss = loss

    def predict(self, *args, **kwargs):
        if not self._use_cuda:
            torch_args = [torch.from_numpy(x) for x in args]
            val = self._network.forward(*torch_args, **kwargs).detach().numpy()
        else:
            torch_args = [torch.from_numpy(x).cuda() for x in args]
            val = self._network.forward(*torch_args,
                                        **kwargs).detach().cpu().numpy()

        return val

    def fit(self, *args, **kwargs):
        if self._dropout:
            self._network.train()

        for t in trange(self._n_epochs, disable=self._quiet):
            if self._batch_size > 0:
                batches = minibatch_generator(self._batch_size, *args)
            else:
                batches = [args]

            loss_current = list()
            for batch in tqdm(batches, disable=self._quiet):
                if not self._use_cuda:
                    torch_args = [torch.from_numpy(x) for x in batch]
                else:
                    torch_args = [torch.from_numpy(x).cuda() for x in args]

                x = torch_args[:-self._n_fit_targets]

                y_hat = self._network(*x, **kwargs)

                if isinstance(y_hat, tuple):
                    output_type = y_hat[0].dtype
                else:
                    output_type = y_hat.dtype

                y = [torch.tensor(y_i, dtype=output_type) for y_i
                     in torch_args[-self._n_fit_targets:]]

                if self._use_cuda:
                    y = [y_i.cuda() for y_i in y]

                loss = self._loss(y_hat, *y)
                loss_current.append(loss.item())

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            if not self._quiet:
                tqdm.write('loss at epoch ' + str(t) + ' ' +
                           str(np.mean(loss_current)))

        if self._dropout:
            self._network.eval()

    def set_weights(self, weights):

        idx = 0
        for p in self._network.parameters():
            shape = p.data.shape

            c = 1
            for s in shape:
                c *= s

            w = np.reshape(weights[idx:idx+c], shape)

            if not self._use_cuda:
                w_tensor = torch.from_numpy(w).type(p.data.dtype)
            else:
                w_tensor = torch.from_numpy(w).type(p.data.dtype).cuda()

            p.data = w_tensor
            idx += c

    def get_weights(self):
        weights = list()

        for p in self._network.parameters():
            w = p.data.detach().cpu().numpy()
            weights.append(w.flatten())

        weights = np.concatenate(weights, 0)

        return weights

    @property
    def weights_size(self):
        return sum(p.numel() for p in self._network.parameters())

    def diff(self, *args, **kwargs):
        if not self._use_cuda:
            torch_args = [torch.from_numpy(np.atleast_2d(x)) for x in args]
        else:
            torch_args = [torch.from_numpy(np.atleast_2d(x)).cuda()
                          for x in args]

        y_hat = self._network(*torch_args, **kwargs)

        gradients = list()
        for i in range(y_hat.shape[1]):
            y_hat[:, i].backward(retain_graph=True)

            gradient = list()
            for p in self._network.parameters():
                g = p.grad.data.detach().cpu().numpy()
                gradient.append(g.flatten())

            g = np.concatenate(gradient, 0)

            gradients.append(g)

        g = np.stack(gradients, -1)

        return g
