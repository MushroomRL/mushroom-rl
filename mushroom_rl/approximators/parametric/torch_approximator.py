import torch
import numpy as np
from tqdm import trange, tqdm

from mushroom_rl.core import Serializable
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import get_weights, set_weights, zero_grad, update_optimizer_parameters


class TorchApproximator(Serializable):
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
                initialized at every fit call. To perform the initialization, 
                the weights_init method must be defined properly for the 
                selected model network.
            dropout (bool, False): if True, dropout is applied only during
                train;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            **params: dictionary of parameters needed to construct the
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

        self._add_save_attr(
            _batch_size='primitive',
            _reinitialize='primitive',
            _use_cuda='primitive',
            _dropout='primitive',
            _quiet='primitive',
            _n_fit_targets='primitive',
            network='torch',
            _optimizer='torch',
            _loss='pickle',
            _last_loss='none'
        )

        self._last_loss = None

    def predict(self, *args, output_tensor=False, **kwargs):
        """
        Predict.

        Args:
            *args: input;
            output_tensor (bool, False): whether to return the output as tensor
                or not;
            **kwargs: other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        if not self._use_cuda:
            torch_args = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                          for x in args]
            val = self.network(*torch_args, **kwargs)

            if output_tensor:
                return val
            elif isinstance(val, tuple):
                val = tuple([x.detach().numpy() for x in val])
            else:
                val = val.detach().numpy()
        else:
            torch_args = [torch.from_numpy(x).cuda()
                          if isinstance(x, np.ndarray) else x.cuda() for x in args]
            val = self.network(*torch_args, **kwargs)

            if output_tensor:
                return val
            elif isinstance(val, tuple):
                val = tuple([x.detach().cpu().numpy() for x in val])
            else:
                val = val.detach().cpu().numpy()

        return val

    def fit(self, *args, n_epochs=None, weights=None, epsilon=None, patience=1,
            validation_split=1., **kwargs):
        """
        Fit the model.

        Args:
            *args: input, where the last ``n_fit_targets`` elements
                are considered as the target, while the others are considered
                as input;
            n_epochs (int, None): the number of training epochs;
            weights (np.ndarray, None): the weights of each sample in the
                computation of the loss;
            epsilon (float, None): the coefficient used for early stopping;
            patience (float, 1.): the number of epochs to wait until stop
                the learning if not improving;
            validation_split (float, 1.): the percentage of the dataset to use
                as training set;
            **kwargs: other parameters used by the fit method of the
                regressor.

        """
        if self._reinitialize:
            self.network.weights_init()

        if self._dropout:
            self.network.train()

        if epsilon is not None:
            n_epochs = np.inf if n_epochs is None else n_epochs
            check_loss = True
        else:
            n_epochs = 1 if n_epochs is None else n_epochs
            check_loss = False

        if weights is not None:
            args += (weights,)
            use_weights = True
        else:
            use_weights = False

        if 0 < validation_split <= 1:
            train_len = np.ceil(len(args[0]) * validation_split).astype(int)
            train_args = [a[:train_len] for a in args]
            val_args = [a[train_len:] for a in args]
        else:
            raise ValueError

        patience_count = 0
        best_loss = np.inf
        epochs_count = 0
        if check_loss:
            with tqdm(total=n_epochs if n_epochs < np.inf else None,
                      dynamic_ncols=True, disable=self._quiet,
                      leave=False) as t_epochs:
                while patience_count < patience and epochs_count < n_epochs:
                    mean_loss_current = self._fit_epoch(train_args, use_weights,
                                                        kwargs)

                    if len(val_args[0]):
                        mean_val_loss_current = self._compute_batch_loss(
                            val_args, use_weights, kwargs
                        )

                        loss = mean_val_loss_current.item()
                    else:
                        loss = mean_loss_current

                    if not self._quiet:
                        t_epochs.set_postfix(loss=loss)
                        t_epochs.update(1)

                    if best_loss - loss > epsilon:
                        patience_count = 0
                        best_loss = loss
                    else:
                        patience_count += 1

                    self._last_loss = mean_loss_current

                    epochs_count += 1
        else:
            with trange(n_epochs, disable=self._quiet) as t_epochs:
                for _ in t_epochs:
                    mean_loss_current = self._fit_epoch(train_args, use_weights,
                                                        kwargs)

                    if not self._quiet:
                        t_epochs.set_postfix(loss=mean_loss_current)

                    self._last_loss = mean_loss_current

        if self._dropout:
            self.network.eval()

    def _fit_epoch(self, args, use_weights, kwargs):
        if self._batch_size > 0:
            batches = minibatch_generator(self._batch_size, *args)
        else:
            batches = [args]

        loss_current = list()
        for batch in batches:
            loss_current.append(self._fit_batch(batch, use_weights, kwargs))

        mean_loss_current = np.mean(loss_current)

        return mean_loss_current

    def _fit_batch(self, batch, use_weights, kwargs):
        loss = self._compute_batch_loss(batch, use_weights, kwargs)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _compute_batch_loss(self, batch, use_weights, kwargs):
        if use_weights:
            weights = torch.from_numpy(batch[-1]).type(torch.float)
            if self._use_cuda:
                weights = weights.cuda()
            batch = batch[:-1]

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

        y = [y_i.clone().detach().type(output_type) for y_i
             in torch_args[-self._n_fit_targets:]]

        if self._use_cuda:
            y = [y_i.cuda() for y_i in y]

        if not use_weights:
            loss = self._loss(y_hat, *y)
        else:
            loss = self._loss(y_hat, *y, reduction='none')
            loss @= weights
            loss = loss / weights.sum()

        return loss

    def set_weights(self, weights):
        """
        Setter.

        Args:
            w (np.ndarray): the set of weights to set.

        """
        set_weights(self.network.parameters(), weights, self._use_cuda)

    def get_weights(self):
        """
        Getter.

        Returns:
            The set of weights of the approximator.

        """
        return get_weights(self.network.parameters())

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return sum(p.numel() for p in self.network.parameters())

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        if not self._use_cuda:
            torch_args = [torch.from_numpy(np.atleast_2d(x)) for x in args]
        else:
            torch_args = [torch.from_numpy(np.atleast_2d(x)).cuda()
                          for x in args]

        y_hat = self.network(*torch_args, **kwargs)
        n_outs = 1 if len(y_hat.shape) == 0 else y_hat.shape[-1]
        y_hat = y_hat.view(-1, n_outs)

        gradients = list()
        for i in range(y_hat.shape[1]):
            zero_grad(self.network.parameters())
            y_hat[:, i].backward(retain_graph=True)

            gradient = list()
            for p in self.network.parameters():
                g = p.grad.data.detach().cpu().numpy()
                gradient.append(g.flatten())

            g = np.concatenate(gradient, 0)

            gradients.append(g)

        g = np.stack(gradients, -1)

        return g

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def loss_fit(self):
        """
        Returns:
            The average loss of the last epoch of the last fit call.
            
        """
        return self._last_loss

    def _post_load(self):
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, list(self.network.parameters()))
