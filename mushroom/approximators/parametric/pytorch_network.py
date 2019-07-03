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
                 loss=None, batch_size=0, n_fit_targets=1, use_cuda=False,
                 reinitialize=False, dropout=False, quiet=True, **params):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input of the network;
            output_shape (tuple): shape of the output of the network;
            network (torch.nn.Module): the network class to use;
            optimizer (dict): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the
                fit method;
            batch_size (int, 0): the size of each minibatch. If 0, the whole
                dataset is fed to the optimizer at each epoch;
            n_fit_targets (int, 1): the number of fit targets used by the fit
                method of the network;
            use_cuda (bool, False): if True, runs the network on the GPU;
            reinitialize (bool, False): if True, the approximator is re
            initialized at every fit call. To perform the initialization, the
            weights_init method must be defined properly for the selected
            model network.
            dropout (bool, False): if True, dropout is applied only during
                train;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            params (dict): dictionary of parameters needed to construct the
                network.

        """
        self._batch_size = batch_size
        self._reinitialize = reinitialize
        self._use_cuda = use_cuda
        self._dropout = dropout
        self._quiet = quiet
        self._n_fit_targets = n_fit_targets

        self.network = network(input_shape, output_shape, use_cuda=use_cuda,
                               dropout=dropout, **params)

        if self._use_cuda:
            self.network.cuda()
        if self._dropout:
            self.network.eval()

        if optimizer is not None:
            self._optimizer = optimizer['class'](self.network.parameters(),
                                                 **optimizer['params'])
        self._loss = loss

    def predict(self, *args, **kwargs):
        if not self._use_cuda:
            torch_args = [torch.from_numpy(x) for x in args]
            val = self.network.forward(*torch_args, **kwargs)
            if isinstance(val, tuple):
                val = tuple([x.detach().numpy() for x in val])
            else:
                val = val.detach().numpy()
        else:
            torch_args = [torch.from_numpy(x).cuda() for x in args]
            val = self.network.forward(*torch_args,
                                       **kwargs)
            if isinstance(val, tuple):
                val = tuple([x.detach().cpu().numpy() for x in val])
            else:
                val = val.detach().cpu().numpy()

        return val

    def fit(self, *args, **kwargs):
        if self._reinitialize:
            self.network.weights_init()

        if self._dropout:
            self.network.train()

        if 'epsilon' in kwargs:
            epsilon = kwargs.pop('epsilon')
            patience = kwargs.pop('patience', 1)
            n_epochs = kwargs.pop('n_epochs', np.inf)
            check_loss = True
        else:
            n_epochs = kwargs.pop('n_epochs', 1)
            check_loss = False

        patience_count = 0
        best_loss = np.inf
        epochs_count = 0
        if check_loss:
            with tqdm(total=n_epochs if n_epochs < np.inf else None,
                      dynamic_ncols=True, disable=self._quiet,
                      leave=False) as t_epochs:
                while patience_count < patience and epochs_count < n_epochs:
                    mean_loss_current = self._fit_epoch(args, kwargs)

                    if not self._quiet:
                        t_epochs.set_postfix(loss=mean_loss_current)
                        t_epochs.update(1)

                    if best_loss - mean_loss_current > epsilon:
                        patience_count = 0
                        best_loss = mean_loss_current
                    else:
                        patience_count += 1

                    epochs_count += 1
        else:
            with trange(n_epochs, disable=self._quiet) as t_epochs:
                for _ in t_epochs:
                    mean_loss_current = self._fit_epoch(args, kwargs)

                    if not self._quiet:
                        t_epochs.set_postfix(loss=mean_loss_current)

        if self._dropout:
            self.network.eval()

    def _fit_epoch(self, args, kwargs):
        if self._batch_size > 0:
            batches = minibatch_generator(self._batch_size, *args)
        else:
            batches = [args]

        loss_current = list()
        for batch in batches:
            loss_current.append(self._fit_batch(batch, kwargs))

        return np.mean(loss_current)

    def _fit_batch(self, batch, kwargs):
        if not self._use_cuda:
            torch_args = [torch.from_numpy(x) for x in batch]
        else:
            torch_args = [torch.from_numpy(x).cuda() for x in batch]

        x = torch_args[:-self._n_fit_targets]

        y_hat = self.network(*x, **kwargs)

        if isinstance(y_hat, tuple):
            output_type = y_hat[0].dtype
        else:
            output_type = y_hat.dtype

        y = [y_i.clone().detach().requires_grad_(False).type(output_type) for y_i
             in torch_args[-self._n_fit_targets:]]

        if self._use_cuda:
            y = [y_i.cuda() for y_i in y]

        loss = self._loss(y_hat, *y)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def set_weights(self, weights):
        idx = 0
        for p in self.network.parameters():
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

        assert idx == weights.size

    def get_weights(self):
        weights = list()

        for p in self.network.parameters():
            w = p.data.detach().cpu().numpy()
            weights.append(w.flatten())

        weights = np.concatenate(weights, 0)

        return weights

    @property
    def weights_size(self):
        return sum(p.numel() for p in self.network.parameters())

    def diff(self, *args, **kwargs):
        if not self._use_cuda:
            torch_args = [torch.from_numpy(np.atleast_2d(x)) for x in args]
        else:
            torch_args = [torch.from_numpy(np.atleast_2d(x)).cuda()
                          for x in args]

        y_hat = self.network(*torch_args, **kwargs)

        gradients = list()
        for i in range(y_hat.shape[1]):
            y_hat[:, i].backward(retain_graph=True)

            gradient = list()
            for p in self.network.parameters():
                g = p.grad.data.detach().cpu().numpy()
                gradient.append(g.flatten())

            g = np.concatenate(gradient, 0)

            gradients.append(g)

        g = np.stack(gradients, -1)

        return g
