from copy import deepcopy

import torch
import numpy as np
from torch.func import stack_module_state, functional_call, vmap, grad

from mushroom_rl.approximators.approximator import Approximator, Ensemble
from mushroom_rl.utils.minibatches import minibatch_generator, ensemble_minibatch_generator
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.torch_training import TorchTrainer


class TorchApproximator(Approximator):
    """
    Class to interface a pytorch model to the mushroom Regressor interface.
    This class implements all is needed to use a generic pytorch model and train it using a specified optimizer and
    objective function. This class supports also minibatches.
    When ``n_models > 1``, construction dispatches to ``TorchEnsemble``.

    """

    def __new__(cls, input_shape=None, output_shape=None, network=None, optimizer=None, loss=None,
                batch_size=0, n_fit_targets=1, reinitialize=False, dropout=False, quiet=True,
                n_models=None, **params):
        if cls is not TorchApproximator:
            return object.__new__(cls)
        if n_models is not None and n_models > 1:
            instance = object.__new__(TorchEnsemble)
            TorchEnsemble.__init__(instance, input_shape=input_shape, output_shape=output_shape,
                                   network=network, optimizer=optimizer, loss=loss,
                                   batch_size=batch_size, n_fit_targets=n_fit_targets,
                                   reinitialize=reinitialize, dropout=dropout, quiet=quiet,
                                   n_models=n_models, **params)
            return instance
        return object.__new__(cls)

    def __init__(self, input_shape, output_shape, network, optimizer=None, loss=None, batch_size=0,
                 n_fit_targets=1, reinitialize=False, dropout=False, quiet=True, n_models=None,
                 n_outputs=1, **params):
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
            reinitialize (bool, False): if True, the approximator is
                reinitialized at every fit call. To perform the initialization,
                the weights_init method must be defined properly for the
                selected model network;
            dropout (bool, False): if True, dropout is applied only during
                train;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            **params: dictionary of parameters needed to construct the
                network.

        """
        super().__init__(backend='torch')

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._parse_output = self._parse_single_output if n_outputs == 1 else self._parse_multi_output

        self.network = network(input_shape, output_shape, dropout=dropout, **params)
        self.network.to(TorchUtils.get_device())

        if dropout:
            self.network.eval()

        if optimizer is not None:
            self._optimizer = optimizer['class'](self.network.parameters(), **optimizer['params'])
        else:
            self._optimizer = None
        self._loss = loss
        self._last_loss = None
        self._dirty = False
        self._trainer = TorchTrainer(loss, batch_size, n_fit_targets, reinitialize, dropout,
                                     self._fit_epoch, self._compute_val_loss, self._store_loss, quiet)

        self._add_save_attr(
            _input_shape='primitive',
            _output_shape='primitive',
            _parse_output='primitive',
            network='torch',
            _optimizer='torch',
            _loss='pickle',
            _last_loss='none',
            _trainer='mushroom',
        )

    def _post_load(self):
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, list(self.network.parameters()))
        self._trainer.set_callbacks(self._fit_epoch, self._compute_val_loss, self._store_loss)

    @staticmethod
    def _parse_single_output(out):
        return out.squeeze(0)

    @staticmethod
    def _parse_multi_output(out):
        return tuple(o.squeeze(0) for o in out)

    def predict(self, *args, **kwargs):
        """
        Predict.

        Args:
            *args: input;
            **kwargs: other parameters used by the predict method of the regressor.

        Returns:
            The predictions of the model.

        """
        if args[0].ndim == len(self._input_shape):
            args = [x.unsqueeze(0) for x in args]
        return self._parse_output(self.network(*args, **kwargs))

    def fit(self, *args, n_epochs=None, weights=None, epsilon=None, patience=1, validation_split=1., **kwargs):
        """
        Fit the model.

        Args:
            *args: input, where the last ``n_fit_targets`` elements are considered as the target,
                while the others are considered as input;
            n_epochs (int, None): the number of training epochs;
            weights (np.ndarray, None): the weights of each sample in the computation of the loss;
            epsilon (float, None): the coefficient used for early stopping;
            patience (float, 1.): the number of epochs to wait until stop the learning if not improving;
            validation_split (float, 1.): the percentage of the dataset to use as training set;
            **kwargs: other parameters used by the fit method of the regressor.

        """
        if self._trainer.reinitialize:
            self.network.weights_init()

        if self._trainer.dropout:
            self.network.train()

        self._trainer.fit(args, n_epochs, weights, epsilon, patience, validation_split, kwargs)
        self._dirty = True

        if self._trainer.dropout:
            self.network.eval()

        self._log()

    def _fit_epoch(self, args, use_weights, network_kwargs):
        if self._trainer.batch_size > 0:
            batches = minibatch_generator(self._trainer.batch_size, *args)
        else:
            batches = [args]

        loss_current = list()
        for batch in batches:
            loss_current.append(self._fit_batch(batch, use_weights, network_kwargs))

        return np.mean(loss_current)

    def _fit_batch(self, batch, use_weights, network_kwargs):
        loss = self._compute_batch_loss(batch, use_weights, network_kwargs)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _compute_batch_loss(self, batch, use_weights, network_kwargs):
        if use_weights:
            weights = torch.as_tensor(batch[-1], device=TorchUtils.get_device()).float()
            batch = batch[:-1]
        else:
            weights = None

        torch_args = [torch.as_tensor(x, device=TorchUtils.get_device()) for x in batch]
        x = torch_args[:-self._trainer.n_fit_targets]
        y_hat = self.network(*x, **network_kwargs)
        y = [y_i.clone().detach().to(TorchUtils.get_device()) for y_i in torch_args[-self._trainer.n_fit_targets:]]

        return self._trainer.compute_loss_from_output(y_hat, y, weights)

    def _compute_val_loss(self, val_args, use_weights, network_kwargs):
        return self._compute_batch_loss(val_args, use_weights, network_kwargs).item()

    def _store_loss(self, loss):
        self._last_loss = loss

    def parameters(self):
        """
        Returns:
            The list of parameters of the network.

        """
        return self.network.parameters()

    @property
    def output_shape(self):
        """
        Returns:
            The shape of the output of the approximator.

        """
        return self._output_shape

    @property
    def loss_fit(self):
        """
        Returns:
            The average loss of the last epoch of the last fit call.

        """
        return self._last_loss

    def set_learning_rate(self, lr):
        """
        Set the learning rate of the optimizer.

        Args:
            lr (float): the new learning rate.

        """
        assert self._optimizer is not None, "Cannot set learning rate: optimizer not set."
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the set of weights to set.

        """
        TorchUtils.set_weights(self.network.parameters(), weights)
        self._dirty = True

    def get_weights(self):
        """
        Getter.

        Returns:
            The set of weights of the approximator.

        """
        return TorchUtils.get_weights(self.network.parameters())

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return sum(p.numel() for p in self.network.parameters())

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action`` if provided.

        Args:
            *args: input;
            **kwargs: other parameters used by the diff method of the regressor.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action`` if provided.

        """
        torch_args = [torch.atleast_2d(x) for x in args]

        y_hat = self.network(*torch_args, **kwargs)
        n_outs = 1 if len(y_hat.shape) == 0 else y_hat.shape[-1]
        y_hat = y_hat.view(-1, n_outs)

        gradients = list()
        for i in range(y_hat.shape[1]):
            TorchUtils.zero_grad(self.network.parameters())
            y_hat[:, i].backward(retain_graph=True)

            gradient = list()
            for p in self.network.parameters():
                g = p.grad.data.detach()
                gradient.append(g.flatten())

            g = torch.concatenate(gradient)
            gradients.append(g)

        return torch.stack(gradients, -1)


class TorchEnsemble(Ensemble):
    """
    Ensemble of TorchApproximator models trained in parallel using ``torch.func.vmap`` and
    ``torch.func.grad``. Constructed automatically by ``TorchApproximator`` when ``n_models > 1``.

    """

    def __init__(self, input_shape, output_shape, network, optimizer=None, loss=None, batch_size=0,
                 n_fit_targets=1, reinitialize=False, dropout=False, quiet=True, n_models=None,
                 prediction=None, **params):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input of the network;
            output_shape (tuple): shape of the output of the network;
            network (torch.nn.Module): the network class to use;
            optimizer (dict): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the fit method;
            batch_size (int, 0): the size of each minibatch. If 0, the whole dataset is fed to
                the optimizer at each epoch;
            n_fit_targets (int, 1): the number of fit targets used by the fit method of the network;
            reinitialize (bool, False): if True, the approximator is reinitialized at every fit call;
            dropout (bool, False): if True, dropout is applied only during train;
            quiet (bool, True): if False, shows a progress bar over epochs;
            n_models (int): number of models in the ensemble;
            prediction (str, None): how to aggregate predictions across models. One of
                ``'mean'``, ``'min'``, ``'max'``, ``'sum'``, or ``None`` to return all predictions;
            **params: dictionary of parameters needed to construct the network.

        """
        super().__init__(TorchApproximator, n_models, prediction=prediction, backend='torch',
                         input_shape=input_shape, output_shape=output_shape, network=network,
                         optimizer=optimizer, loss=loss, batch_size=batch_size,
                         n_fit_targets=n_fit_targets, reinitialize=reinitialize,
                         dropout=dropout, quiet=quiet, **params)

        self._base_model = deepcopy(self._models[0].network).to('meta').eval()
        self._params, self._buffers = stack_module_state([m.network for m in self._models])
        self._trainer = TorchTrainer(loss, batch_size, n_fit_targets, reinitialize, dropout,
                                     self._fit_epoch, self._compute_val_loss,
                                     self._store_loss, quiet)

        self._add_save_attr(_trainer='mushroom')

    def _sync_params(self):
        if any(m._dirty for m in self._models):
            self._params, self._buffers = stack_module_state([m.network for m in self._models])
            for m in self._models:
                m._dirty = False

    def _post_load(self):
        self._base_model = deepcopy(self._models[0].network).to('meta').eval()
        for m in self._models:
            m._dirty = True
        self._sync_params()
        self._trainer.set_callbacks(self._fit_epoch, self._compute_val_loss, self._store_loss)

    def predict(self, *args, idx=None, prediction=None, **kwargs):
        """
        Predict.

        Args:
            *args: input;
            idx (int, None): if provided, use only the model at that index;
            prediction (str, None): aggregation mode, overrides the constructor default.
                One of ``'mean'``, ``'min'``, ``'max'``, ``'sum'``, or ``None`` to return all;
            **kwargs: other parameters used by the predict method of the regressor.

        Returns:
            The predictions of the model, aggregated according to ``prediction``.

        """
        if idx is not None:
            return self._models[idx].predict(*args, **kwargs)

        self._sync_params()
        torch_args = tuple(torch.atleast_2d(torch.as_tensor(x, device=TorchUtils.get_device())) for x in args)

        def fwd(params, buffers):
            return functional_call(self._base_model, (params, buffers), torch_args, kwargs=kwargs)

        predictions = vmap(fwd)(self._params, self._buffers)

        prediction = prediction if prediction is not None else self._prediction
        if prediction is None:
            return predictions
        if prediction == 'mean':
            return predictions.mean(0)
        elif prediction == 'min':
            return predictions.min(0).values
        elif prediction == 'max':
            return predictions.max(0).values
        elif prediction == 'sum':
            return predictions.sum(0)
        raise ValueError

    def fit(self, *args, idx=None, n_epochs=None, weights=None, epsilon=None, patience=1,
            validation_split=1., **kwargs):
        """
        Fit the model.

        Args:
            *args: input, where the last ``n_fit_targets`` elements are considered as the target,
                while the others are considered as input;
            idx (int, None): if provided, fit only the model at that index;
            n_epochs (int, None): the number of training epochs;
            weights (np.ndarray, None): the weights of each sample in the computation of the loss;
            epsilon (float, None): the coefficient used for early stopping;
            patience (float, 1.): the number of epochs to wait until stop the learning if not improving;
            validation_split (float, 1.): the percentage of the dataset to use as training set;
            **kwargs: other parameters used by the fit method of the regressor.

        """
        if idx is not None:
            self._models[idx].fit(*args, n_epochs=n_epochs, weights=weights, epsilon=epsilon,
                                  patience=patience, validation_split=validation_split, **kwargs)
            self._sync_params()
            self._log()
            return

        if self._trainer.reinitialize:
            for m in self._models:
                m.network.weights_init()

        if self._trainer.dropout:
            self._base_model.train()
            for m in self._models:
                m.network.train()

        self._trainer.fit(args, n_epochs, weights, epsilon, patience, validation_split, kwargs)

        if self._trainer.dropout:
            self._base_model.eval()
            for m in self._models:
                m.network.eval()

        self._sync_params()
        self._log()

    def _fit_epoch(self, args, use_weights, network_kwargs):
        n_models = len(self._models)
        if self._trainer.batch_size > 0:
            batches = ensemble_minibatch_generator(self._trainer.batch_size, n_models, *args)
        else:
            batches = [[torch.as_tensor(a, device=TorchUtils.get_device()).unsqueeze(0).expand(n_models, *a.shape) for a in args]]

        loss_current = []
        for batch in batches:
            loss_current.append(self._fit_batch(batch, use_weights, network_kwargs))

        return np.mean(loss_current, axis=0)

    def _fit_batch(self, stacked_batch, use_weights, network_kwargs):
        self._sync_params()

        if use_weights:
            stacked_w = stacked_batch[-1].float()
            stacked_data = stacked_batch[:-1]
        else:
            stacked_w = None
            stacked_data = stacked_batch

        stacked_x = tuple(stacked_data[:-self._trainer.n_fit_targets])
        stacked_y = [yi.clone().detach() for yi in stacked_data[-self._trainer.n_fit_targets:]]

        base = self._base_model
        nx = len(stacked_x)
        ny = len(stacked_y)
        trainer = self._trainer

        def compute_loss(params, buffers, *data_w):
            x_in = data_w[:nx]
            y_in = data_w[nx:nx + ny]
            w = data_w[-1] if stacked_w is not None else None
            y_hat = functional_call(base, (params, buffers), x_in, kwargs=network_kwargs)
            loss = trainer.compute_loss_from_output(y_hat, list(y_in), w)
            return loss, loss

        all_data = stacked_x + tuple(stacked_y)
        if stacked_w is not None:
            all_data = all_data + (stacked_w,)

        per_model_grads, per_model_losses = vmap(
            grad(compute_loss, has_aux=True)
        )(self._params, self._buffers, *all_data)

        for i, m in enumerate(self._models):
            m._optimizer.zero_grad()
            for name, param in m.network.named_parameters():
                param.grad = per_model_grads[name][i]
            m._optimizer.step()
            m._dirty = True

        return per_model_losses.detach().cpu().numpy()

    def _compute_val_loss(self, val_args, use_weights, network_kwargs):
        if use_weights:
            weights = torch.as_tensor(val_args[-1], device=TorchUtils.get_device()).float()
            val_args = val_args[:-1]
        else:
            weights = None

        torch_batch = [torch.as_tensor(x, device=TorchUtils.get_device()) for x in val_args]
        x_inputs = tuple(torch_batch[:-self._trainer.n_fit_targets])
        y_targets = [yi.clone().detach() for yi in torch_batch[-self._trainer.n_fit_targets:]]

        base = self._base_model
        trainer = self._trainer

        def compute_loss(params, buffers):
            y_hat = functional_call(base, (params, buffers), x_inputs, kwargs=network_kwargs)
            return trainer.compute_loss_from_output(y_hat, y_targets, weights)

        with torch.no_grad():
            losses = vmap(compute_loss)(self._params, self._buffers)
        return float(losses.mean().item())

    def _store_loss(self, losses):
        for m, ml in zip(self._models, losses):
            m._last_loss = float(ml)

    def parameters(self):
        """
        Returns:
            The concatenated parameters of all models in the ensemble.

        """
        return [p for m in self._models for p in m.parameters()]

    @property
    def network(self):
        """
        Returns:
            The network of the first model in the ensemble.

        """
        return self._models[0].network

    @property
    def loss_fit(self):
        """
        Returns:
            List of per-model losses from the last fit call.

        """
        return [m.loss_fit for m in self._models]

    def set_learning_rate(self, lr):
        """
        Set the learning rate of the optimizer of all models in the ensemble.

        Args:
            lr (float): the new learning rate.

        """
        for m in self._models:
            m.set_learning_rate(lr)

    def set_weights(self, weights):
        """
        Set weights for each model in the ensemble independently.

        Args:
            weights: tensor of shape ``(n_models, weights_size_per_model)``.

        """
        for i, m in enumerate(self._models):
            m.set_weights(weights[i])
        self._sync_params()

    def get_weights(self):
        """
        Returns:
            The stacked weights of all models, shape ``(n_models, weights_size_per_model)``.

        """
        return torch.stack([m.get_weights() for m in self._models])

    @property
    def weights_size(self):
        """
        Returns:
            The shape of the stacked weights matrix ``(n_models, weights_size_per_model)``.

        """
        return len(self._models), self._models[0].weights_size

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. the input for each model, stacked.

        Args:
            *args: input;
            **kwargs: other parameters used by the diff method of the regressor.

        Returns:
            The stacked derivatives of all models w.r.t. the input, shape ``(n_models, weights_size, n_outputs)``.

        """
        return torch.stack([m.diff(*args, **kwargs) for m in self._models], dim=0)


class NumpyTorchApproximator(TorchApproximator):
    """
    Wrapper to get a Numpy interface to the TorchApproximator class.
    This class allows you to use the torch approximator with numpy backend algorithms.

    """
    def predict(self, *args, **kwargs):
        torch_args = [torch.as_tensor(x, device=TorchUtils.get_device()) for x in args]
        return super().predict(*torch_args, **kwargs).detach().cpu().numpy()

    def fit(self, *args, n_epochs=None, weights=None, epsilon=None, patience=1, validation_split=1., **kwargs):
        torch_args = [torch.as_tensor(x, device=TorchUtils.get_device()) for x in args]
        super().fit(*torch_args, n_epochs=n_epochs, weights=weights, epsilon=epsilon, patience=patience,
                    validation_split=validation_split, **kwargs)

    def diff(self, *args, **kwargs):
        torch_args = [torch.as_tensor(np.atleast_2d(x), device=TorchUtils.get_device()) for x in args]
        gradient = super().diff(*torch_args, **kwargs)
        return gradient.detach().cpu().numpy()

    def set_weights(self, weights):
        super().set_weights(torch.as_tensor(weights))

    def get_weights(self):
        return super().get_weights().detach().cpu().numpy()