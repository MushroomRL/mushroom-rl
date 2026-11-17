from copy import deepcopy

import torch
from torch.func import stack_module_state, functional_call, vmap, grad

from mushroom_rl.core import MushroomObject
from mushroom_rl.approximators.approximator import Approximator, Ensemble
from mushroom_rl.utils.minibatches import minibatch_generator, ensemble_minibatch_generator
from mushroom_rl.utils.torch_utils import TorchUtils, TorchTrainer


class TorchApproximator(Approximator):
    """
    Class to interface a pytorch model to the mushroom Approximator interface.
    This class implements all is needed to use a generic pytorch model and train it using a specified optimizer and
    objective function. This class supports also minibatches.
    When ``n_models > 1``, construction dispatches to ``TorchEnsemble``.

    """

    def __new__(cls, *args, n_models=None, **kwargs):
        if cls is TorchApproximator and n_models is not None and n_models > 1:
            instance = MushroomObject.__new__(TorchEnsemble)
            TorchEnsemble.__init__(instance, *args, n_models=n_models, **kwargs)
            return instance
        else:
            return MushroomObject.__new__(cls)

    def __init__(self, network, input_shape, output_shape, optimizer=None, loss=None, batch_size=0, n_fit_targets=1,
                 reinitialize=False, dropout=False, quiet=True, n_models=None, action_history_shape=None,
                 kwargs_shape=None, device=None, **params):
        """
        Constructor.

        Args:
            network (torch.nn.Module): the network class to use;
            input_shape (tuple, list): shape of the input of the network. A plain tuple for a
                single-input network, or a list of shape tuples (one per positional input) for a
                network that takes several distinct inputs (e.g. a critic taking ``state`` and
                ``action`` separately);
            output_shape (tuple, list): shape of the output of the network. A plain tuple for a
                single-output network, or a list of shape tuples (one per output tensor) for a
                network whose ``forward`` returns several tensors; the number of outputs to parse
                is derived from this, not passed separately;
            optimizer (dict): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the
                fit method;
            batch_size (int, 0): the size of each minibatch. If 0, the whole
                dataset is fed to the optimizer at each epoch;
            n_fit_targets (int, 1): the number of fit targets used by the fit
                method of the network;
            reinitialize (bool, False): if True, the network and the optimizer are reset at every fit call;
            dropout (bool, False): if True, dropout is applied only during
                train;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            action_history_shape (tuple, None): the per-timestep shape of the ``action_history``
                keyword input, when the network consumes the previous action. Convenience shortcut
                for ``kwargs_shape={'action_history': action_history_shape}``;
            kwargs_shape (dict, None): mapping ``{name: shape}`` declaring keyword inputs of the
                network. Declared keyword inputs are batch-padded by ``predict`` exactly like the
                positional inputs, and each shape is forwarded to the network constructor under the
                ``{name}_shape`` keyword (e.g. ``action_history`` -> ``action_history_shape``);
            device (str, None): device the network is stored on, or ``None`` for the default one;
            **params: dictionary of parameters needed to construct the
                network.

        """
        super().__init__(input_shape=input_shape, output_shape=output_shape, backend='torch', device=device)

        self._input_ndims = [len(s) for s in input_shape] if isinstance(input_shape, list) else [len(input_shape)]

        kwargs_shape = dict(kwargs_shape) if kwargs_shape is not None else dict()
        if action_history_shape is not None:
            kwargs_shape['action_history'] = action_history_shape
        self._kwargs_ndims = {name: len(shape) for name, shape in kwargs_shape.items()}

        network_shape_kwargs = {f'{name}_shape': shape for name, shape in kwargs_shape.items()}
        self.network = network(input_shape, output_shape, dropout=dropout, **network_shape_kwargs, **params)
        self.network.to(self._device)

        if dropout:
            self.network.eval()

        if optimizer is not None:
            self._optimizer = optimizer['class'](self.network.parameters(), **optimizer['params'])
        else:
            self._optimizer = None
        self._last_loss = None
        self._trainer = TorchTrainer(loss, batch_size, n_fit_targets, reinitialize, dropout,
                                     self._fit_epoch, self._compute_val_loss, self._store_loss, quiet,
                                     device=self._device)

        self._add_save_attr(
            _input_ndims='primitive',
            _kwargs_ndims='primitive',
            network='torch',
            _optimizer='torch',
            _last_loss='none',
            _trainer='mushroom',
        )

    def _post_load(self, device):
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, list(self.network.parameters()))
        self._trainer.set_callbacks(self._fit_epoch, self._compute_val_loss, self._store_loss)

    def _reinitialize(self):
        self.network.reset_parameters()
        self._optimizer.state.clear()

    def _pad_inputs(self, args, kwargs):
        n_declared = len(self._input_ndims)
        args = [a.unsqueeze(0) if i < n_declared and a.ndim == self._input_ndims[i] else a
                for i, a in enumerate(args)]
        for name, ndim in self._kwargs_ndims.items():
            value = kwargs[name]
            if value.ndim == ndim:
                kwargs[name] = value.unsqueeze(0)

        return args, kwargs

    def predict(self, *args, **kwargs):
        """
        Predict. The positional inputs and the keyword inputs declared via ``kwargs_shape`` /
        ``action_history_shape`` are batch-padded when a single (unbatched) sample is provided. A declared
        keyword input that is not passed raises a ``KeyError``.

        Args:
            *args: input;
            **kwargs: other parameters used by the predict method of the network, including any declared
                keyword input.

        Returns:
            The predictions of the model.

        """
        single = self._is_single(args[0])
        args, kwargs = self._pad_inputs(args, kwargs)

        return self._unbatch(self.network(*args, **kwargs), single)

    def fit(self, *args, n_epochs=None, weights=None, epsilon=None, patience=1, validation_split=1., **kwargs):
        """
        Fit the model.

        Args:
            *args: input, where the last ``n_fit_targets`` elements are considered as the target,
                while the others are considered as input;
            n_epochs (int, None): the number of training epochs;
            weights (torch.Tensor, None): the weights of each sample in the computation of the loss;
            epsilon (float, None): the coefficient used for early stopping;
            patience (float, 1.): the number of epochs to wait until stop the learning if not improving;
            validation_split (float, 1.): the percentage of the dataset to use as training set;
            **kwargs: other parameters used by the fit method of the approximator, including any keyword input
                declared via ``kwargs_shape``/``action_history_shape``.

        """
        if self._trainer.reinitialize:
            self._reinitialize()

        if self._trainer.dropout:
            self.network.train()

        n_targets = self._trainer.n_fit_targets
        kwargs_data = tuple(kwargs.pop(name) for name in self._kwargs_ndims)
        args = args[:-n_targets] + kwargs_data + args[-n_targets:]

        self._trainer.fit(args, n_epochs, weights, epsilon, patience, validation_split, kwargs)

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
            loss_batch = self._fit_batch(batch, use_weights, network_kwargs)
            loss_current.append(loss_batch)

        return torch.stack(loss_current).mean()

    def _fit_batch(self, batch, use_weights, network_kwargs):
        loss = self._compute_batch_loss(batch, use_weights, network_kwargs)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.detach()

    def _compute_batch_loss(self, batch, use_weights, network_kwargs):
        if use_weights:
            weights = torch.as_tensor(batch[-1], device=self._device).float()
            batch = batch[:-1]
        else:
            weights = None

        n_kwargs = len(self._kwargs_ndims)
        x_end = len(batch) - self._trainer.n_fit_targets - n_kwargs
        x = batch[:x_end]
        kwargs_data = dict(zip(self._kwargs_ndims, batch[x_end:x_end + n_kwargs]))
        y = batch[-self._trainer.n_fit_targets:]

        y_hat = self.network(*x, **network_kwargs, **kwargs_data)
        y = [y_i.clone().detach() for y_i in y]

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
    def loss(self):
        """
        Returns:
            The loss function optimized by the fit method.

        """
        return self._trainer.loss

    @property
    def loss_fit(self):
        """
        Returns:
            The average loss of the last epoch of the last fit call.

        """
        return float(self._last_loss) if self._last_loss is not None else None

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
        Compute the derivative of the output w.r.t. the model parameters for a single sample, given either unbatched
        or as a batch of one. Multi-output networks are not supported.

        Args:
            *args: input;
            **kwargs: other parameters used by the diff method of the approximator.

        Returns:
            The derivative of the output w.r.t. the model parameters.

        """
        assert not isinstance(self._output_shape, list), 'diff is not supported by multi-output networks.'

        n_declared = len(self._input_ndims)
        args, kwargs = self._pad_inputs(args, kwargs)
        args = args[:n_declared] + [torch.atleast_2d(a) for a in args[n_declared:]]

        y_hat = self.network(*args, **kwargs)
        assert y_hat.shape[0] == 1, 'diff supports a single sample only.'

        parameters = list(self.network.parameters())
        gradients = [TorchUtils.compute_gradient(y, parameters, retain_graph=True) for y in y_hat.flatten()]

        return torch.stack(gradients, -1).reshape(-1, *y_hat.shape[1:])


class TorchEnsembleMember(TorchApproximator):
    """
    Model of a ``TorchEnsemble``. Its parameters are the slice of the ensemble's stacked parameters at its index:
    it is saved, loaded and moved by its ensemble only, and it can be fitted alone only when the ensemble has one
    optimizer per model.

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Args:
            *args: positional parameters of ``TorchApproximator``;
            **kwargs: keyword parameters of ``TorchApproximator``.

        """
        super().__init__(*args, **kwargs)

        self._ensemble_move = False
        del self._save_attributes['network']

    def save(self, path, full_save=False):
        raise NotImplementedError('A model of an ensemble is saved with its ensemble: call save on the ensemble.')

    def to(self, device):
        assert self._ensemble_move, 'A model of an ensemble moves with its ensemble: call to on the ensemble.'
        return super().to(device)

    def fit(self, *args, **kwargs):
        assert self._optimizer is not None, \
            'Fitting a single model of an ensemble requires an ensemble built with shared_optimizer=False.'
        super().fit(*args, **kwargs)

    def _post_load(self, device):
        self._ensemble_move = False
        self._trainer.set_callbacks(self._fit_epoch, self._compute_val_loss, self._store_loss)


class TorchEnsemble(Ensemble):
    """
    Ensemble of TorchApproximator models trained in parallel using ``torch.func.vmap`` and
    ``torch.func.grad``. Constructed automatically by ``TorchApproximator`` when ``n_models > 1``.

    """

    def __init__(self, network, input_shape, output_shape, optimizer=None, loss=None, batch_size=0, n_fit_targets=1,
                 reinitialize=False, dropout=False, quiet=True, n_models=None, prediction='mean',
                 randomness='different', shared_optimizer=True, action_history_shape=None, kwargs_shape=None,
                 device=None, **params):
        """
        Constructor.

        Args:
            network (torch.nn.Module): the network class to use;
            input_shape (tuple, list): shape of the input of the network. A plain tuple for a
                single-input network, or a list of shape tuples (one per positional input) for a
                network that takes several distinct inputs;
            output_shape (tuple, list): shape of the output of the network. A plain tuple for a
                single-output network, or a list of shape tuples (one per output tensor) for a
                network whose ``forward`` returns several tensors;
            optimizer (dict): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the fit method;
            batch_size (int, 0): the size of each minibatch. If 0, the whole dataset is fed to
                the optimizer at each epoch;
            n_fit_targets (int, 1): the number of fit targets used by the fit method of the network;
            reinitialize (bool, False): if True, the networks and the optimizers are reset at every fit call;
            dropout (bool, False): if True, dropout is applied only during train;
            quiet (bool, True): if False, shows a progress bar over epochs;
            n_models (int): number of models in the ensemble;
            prediction (str, 'mean'): how to aggregate predictions across models. One of ``'mean'``,
                ``'min'``, ``'max'``, ``'sum'``, or ``'all'`` to return all predictions;
            randomness (str, 'different'): how random operations inside the network (e.g. the noise of a
                noisy layer, or dropout) behave across the models of the ensemble. ``'different'`` draws
                independent randomness per model, ``'same'`` shares a single draw across all models, and
                ``'error'`` forbids random operations altogether;
            shared_optimizer (bool, True): if True, a single optimizer updates all the models, stepping the
                parameters shared between models once with the sum of their gradients; if False, each model has its
                own optimizer, and a single model can be fitted with ``idx``;
            action_history_shape (tuple, None): the per-timestep shape of the ``action_history``
                keyword input, when the network consumes the previous action. Convenience shortcut
                for ``kwargs_shape={'action_history': action_history_shape}``;
            kwargs_shape (dict, None): mapping ``{name: shape}`` declaring keyword inputs of the
                network. Declared keyword inputs are batch-padded by ``predict`` exactly like the
                positional inputs, and each shape is forwarded to the network constructor under the
                ``{name}_shape`` keyword (e.g. ``action_history`` -> ``action_history_shape``);
            device (str, None): device the networks are stored on, or ``None`` for the default one;
            **params: dictionary of parameters needed to construct the network.

        """
        super().__init__(TorchEnsembleMember, n_models, prediction=prediction,
                         input_shape=input_shape, output_shape=output_shape, network=network,
                         optimizer=None if shared_optimizer else optimizer, loss=loss, batch_size=batch_size,
                         n_fit_targets=n_fit_targets, reinitialize=reinitialize,
                         dropout=dropout, quiet=quiet, action_history_shape=action_history_shape,
                         kwargs_shape=kwargs_shape, device=device, **params)

        self._randomness = randomness
        self._shared_optimizer = shared_optimizer
        self._input_ndims = [len(s) for s in input_shape] if isinstance(input_shape, list) else [len(input_shape)]

        kwargs_shape = dict(kwargs_shape) if kwargs_shape is not None else dict()
        if action_history_shape is not None:
            kwargs_shape['action_history'] = action_history_shape
        self._kwargs_ndims = {name: len(shape) for name, shape in kwargs_shape.items()}

        networks = [m.network for m in self._models]
        params, buffers = stack_module_state(networks)
        self._shared_params = self._shared_names(networks, 'get_parameter', params)
        self._shared_buffers = self._shared_names(networks, 'get_buffer', buffers)
        self._params = {name: torch.nn.Parameter(p.detach(), requires_grad=p.requires_grad)
                        for name, p in params.items() if name not in self._shared_params}
        self._buffers = {name: b for name, b in buffers.items() if name not in self._shared_buffers}
        self._shared_objects = self._find_shared_objects(networks, self._shared_params + self._shared_buffers)
        self._bind_members()

        if shared_optimizer and optimizer is not None:
            self._optimizer = optimizer['class'](self.parameters(), **optimizer['params'])
        else:
            self._optimizer = None

        self._base_model = self._meta_template(networks[0])
        self._trainer = TorchTrainer(loss, batch_size, n_fit_targets, reinitialize, dropout,
                                     self._fit_epoch, self._compute_val_loss,
                                     self._store_loss, quiet, device=self._device)

        self._add_save_attr(
            _randomness='primitive',
            _shared_optimizer='primitive',
            _input_ndims='primitive',
            _kwargs_ndims='primitive',
            _shared_params='primitive',
            _shared_buffers='primitive',
            _base_model='pickle',
            _params='torch',
            _buffers='torch',
            _optimizer='torch',
            _shared_objects='torch',
            _trainer='mushroom',
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._bind_members()

    def to(self, device):
        for m in self._models:
            m._ensemble_move = True

        return super().to(device)

    def predict(self, *args, idx=None, prediction=None, compute_variance=False, **kwargs):
        """
        Predict. The positional inputs and the keyword inputs declared via ``kwargs_shape`` /
        ``action_history_shape`` are batch-padded when a single (unbatched) sample is provided, exactly as by
        a single model. A declared keyword input that is not passed raises a ``KeyError``.

        Args:
            *args: input;
            idx (int, None): if provided, use only the model at that index;
            prediction (str, None): aggregation mode, overrides the constructor default. One of ``'mean'``,
                ``'min'``, ``'max'``, ``'sum'``, or ``'all'`` to return all predictions. ``None`` uses the
                constructor default;
            compute_variance (bool, False): if ``True``, also return the variance across models.
            **kwargs: other parameters used by the predict method of the approximator, including any declared
                keyword input.

        Returns:
            The predictions of the model, aggregated according to ``prediction``. An aggregated prediction has
            the same shape as the prediction of a single model, while an un-aggregated one carries an
            additional leading model axis. If ``compute_variance`` is ``True``, a list
            ``[predictions, variance]`` is returned instead.

        """
        assert not compute_variance or prediction != 'all'
        if idx is not None:
            return self._models[idx].predict(*args, **kwargs)

        n_declared = len(self._input_ndims)
        torch_args = tuple(torch.as_tensor(x, device=self._device) for x in args)
        single = self._is_single(torch_args[0])
        torch_args = tuple(a.unsqueeze(0) if i < n_declared and a.ndim == self._input_ndims[i] else a
                           for i, a in enumerate(torch_args))
        for name, ndim in self._kwargs_ndims.items():
            value = kwargs[name]
            if value.ndim == ndim:
                kwargs[name] = value.unsqueeze(0)

        def fwd(params, buffers):
            return functional_call(self._base_model, (params, buffers), torch_args, kwargs=kwargs)

        prediction = prediction if prediction is not None else self._prediction

        out = self._unbatch(vmap(fwd, randomness=self._randomness)(*self._functional_state()), single, axis=1)

        return self._parse_output(out, prediction, compute_variance)

    def fit(self, *args, idx=None, n_epochs=None, weights=None, epsilon=None, patience=1, validation_split=1.,
            **kwargs):
        """
        Fit the model.

        Args:
            *args: input, where the last ``n_fit_targets`` elements are considered as the target,
                while the others are considered as input;
            idx (int, None): if provided, fit only the model at that index. Requires an ensemble built with
                ``shared_optimizer=False``;
            n_epochs (int, None): the number of training epochs;
            weights (torch.Tensor, None): the weights of each sample in the computation of the loss;
            epsilon (float, None): the coefficient used for early stopping;
            patience (float, 1.): the number of epochs to wait until stop the learning if not improving;
            validation_split (float, 1.): the percentage of the dataset to use as training set;
            **kwargs: other parameters used by the fit method of the approximator, including any keyword input
                declared via ``kwargs_shape``/``action_history_shape``.

        """
        if idx is not None:
            self._models[idx].fit(*args, n_epochs=n_epochs, weights=weights, epsilon=epsilon,
                                  patience=patience, validation_split=validation_split, **kwargs)
            for param in self._params.values():
                param.grad = None
            return

        if self._trainer.reinitialize:
            self._reinitialize()

        if self._trainer.dropout:
            self._base_model.train()
            for m in self._models:
                m.network.train()

        n_targets = self._trainer.n_fit_targets
        kwargs_data = tuple(kwargs.pop(name) for name in self._kwargs_ndims)
        args = args[:-n_targets] + kwargs_data + args[-n_targets:]

        self._trainer.fit(args, n_epochs, weights, epsilon, patience, validation_split, kwargs)

        if self._trainer.dropout:
            self._base_model.eval()
            for m in self._models:
                m.network.eval()

        self._log()

    def parameters(self):
        """
        Returns:
            The parameters of the ensemble: a stacked tensor with a leading model axis for each parameter of the
            network, and the parameters the models share with each other once each.

        """
        return [self._params.get(name, p) for name, p in self._models[0].network.named_parameters()]

    def set_learning_rate(self, lr):
        """
        Set the learning rate of the optimizer of all models in the ensemble.

        Args:
            lr (float): the new learning rate.

        """
        if self._shared_optimizer:
            assert self._optimizer is not None, "Cannot set learning rate: optimizer not set."
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for m in self._models:
                m.set_learning_rate(lr)

    def set_weights(self, weights):
        """
        Set weights for each model in the ensemble independently.

        Args:
            weights: tensor of shape ``(n_models, weights_size_per_model)``.

        """
        offset = 0
        for name, p in self._models[0].network.named_parameters():
            size = p.numel()
            block = torch.as_tensor(weights[:, offset:offset + size])
            if name in self._params:
                self._params[name].data.copy_(block.reshape(self._params[name].shape))
            else:
                p.data.copy_(block[-1].reshape(p.shape))
            offset += size

    def get_weights(self):
        """
        Returns:
            The stacked weights of all models, shape ``(n_models, weights_size_per_model)``.

        """
        n_models = len(self._models)

        return torch.cat([self._params[name].detach().reshape(n_models, -1) if name in self._params
                          else p.detach().reshape(1, -1).expand(n_models, -1)
                          for name, p in self._models[0].network.named_parameters()], 1)

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. the model parameters for each model, stacked.

        Args:
            *args: input;
            **kwargs: other parameters used by the diff method of the approximator.

        Returns:
            The stacked derivatives of all models w.r.t. the model parameters, shape
            ``(n_models, weights_size, *output_shape)``.

        """
        return torch.stack([m.diff(*args, **kwargs) for m in self._models], dim=0)

    @property
    def network(self):
        """
        Returns:
            The network of the first model in the ensemble.

        """
        return self._models[0].network

    @property
    def loss(self):
        """
        Returns:
            The loss function optimized by the fit method.

        """
        return self._trainer.loss

    @property
    def loss_fit(self):
        """
        Returns:
            List of per-model losses from the last fit call.

        """
        return [m.loss_fit for m in self._models]

    @property
    def weights_size(self):
        """
        Returns:
            The shape of the stacked weights matrix ``(n_models, weights_size_per_model)``.

        """
        return len(self._models), self._models[0].weights_size

    def _post_load(self, device):
        super()._post_load(device)

        if hasattr(self._models[0], 'network'):
            self._bind_members()
        else:
            self._build_members()
        if self._optimizer is not None:
            TorchUtils.update_optimizer_parameters(self._optimizer, self.parameters())
        for m in self._models:
            if m._optimizer is not None:
                TorchUtils.update_optimizer_parameters(m._optimizer, list(m.network.parameters()))
        self._trainer.set_callbacks(self._fit_epoch, self._compute_val_loss, self._store_loss)

    def _build_members(self):
        template = self._base_model
        for i, m in enumerate(self._models):
            memo = {id(self._get_object(template, name)): obj for name, obj in self._shared_objects.items()}
            memo.update({id(template.get_parameter(name)): torch.nn.Parameter(p.data[i], requires_grad=p.requires_grad)
                         for name, p in self._params.items()})
            memo.update({id(template.get_buffer(name)): b[i] for name, b in self._buffers.items()})
            m.network = deepcopy(template, memo)

    def _bind_members(self):
        for i, network in enumerate(m.network for m in self._models):
            for name, obj in self._shared_objects.items():
                module_name, _, attribute = name.rpartition('.')
                setattr(network.get_submodule(module_name), attribute, obj)
            for name, p in self._params.items():
                param = network.get_parameter(name)
                param.data = p.data[i]
                if param.grad is not None:
                    param.grad = param.grad.to(p.device)
            for name, b in self._buffers.items():
                network.get_buffer(name).data = b.data[i]

    def _functional_state(self):
        params, buffers = self._params, self._buffers
        if self._shared_params or self._shared_buffers:
            n_models = len(self._models)
            network = self._models[0].network
            params = dict(params)
            buffers = dict(buffers)
            for name in self._shared_params:
                p = network.get_parameter(name)
                params[name] = p.expand(n_models, *p.shape)
            for name in self._shared_buffers:
                b = network.get_buffer(name)
                # a copy per model: in-place updates made by the forward cannot be written to a single shared buffer
                buffers[name] = b.expand(n_models, *b.shape).clone()

        return params, buffers

    def _reinitialize(self):
        if self._shared_optimizer:
            for m in self._models:
                m.network.reset_parameters()
            self._optimizer.state.clear()
        else:
            for m in self._models:
                m._reinitialize()

    def _parse_single_output(self, out, prediction, compute_variance):
        return self._aggregate(out, prediction, compute_variance)

    def _parse_multi_output(self, out, prediction, compute_variance):
        return tuple(self._aggregate(o, prediction, compute_variance) for o in out)

    def _fit_epoch(self, args, use_weights, network_kwargs):
        n_models = len(self._models)
        if self._trainer.batch_size > 0:
            batches = ensemble_minibatch_generator(self._trainer.batch_size, n_models, *args)
        else:
            batches = [[torch.as_tensor(a, device=self._device).unsqueeze(0).expand(n_models, *a.shape)
                        for a in args]]

        loss_current = []
        for batch in batches:
            loss_current.append(self._fit_batch(batch, use_weights, network_kwargs))

        return torch.stack(loss_current).mean(dim=0)

    def _fit_batch(self, stacked_batch, use_weights, network_kwargs):
        if use_weights:
            stacked_w = stacked_batch[-1].float()
            stacked_data = stacked_batch[:-1]
        else:
            stacked_w = None
            stacked_data = stacked_batch

        n_kwargs = len(self._kwargs_ndims)
        x_end = len(stacked_data) - self._trainer.n_fit_targets - n_kwargs
        stacked_x = tuple(stacked_data[:x_end])
        stacked_kwargs_data = tuple(stacked_data[x_end:x_end + n_kwargs])
        stacked_y = [yi.clone().detach() for yi in stacked_data[-self._trainer.n_fit_targets:]]

        base = self._base_model
        nx = len(stacked_x)
        nk = len(stacked_kwargs_data)
        ny = len(stacked_y)
        kwargs_names = tuple(self._kwargs_ndims)
        trainer = self._trainer

        def compute_loss(params, buffers, *data_w):
            x_in = data_w[:nx]
            k_in = data_w[nx:nx + nk]
            y_in = data_w[nx + nk:nx + nk + ny]
            w = data_w[-1] if stacked_w is not None else None
            call_kwargs = {**network_kwargs, **dict(zip(kwargs_names, k_in))}
            y_hat = functional_call(base, (params, buffers), x_in, kwargs=call_kwargs)
            loss = trainer.compute_loss_from_output(y_hat, list(y_in), w)
            return loss, loss

        all_data = stacked_x + stacked_kwargs_data + tuple(stacked_y)
        if stacked_w is not None:
            all_data = all_data + (stacked_w,)

        with torch.no_grad():
            per_model_grads, per_model_losses = vmap(
                grad(compute_loss, has_aux=True), randomness=self._randomness
            )(*self._functional_state(), *all_data)
        self._optimizer_step(per_model_grads)

        return per_model_losses

    def _optimizer_step(self, per_model_grads):
        if self._shared_optimizer:
            self._clear_member_gradients()
            self._optimizer.zero_grad()
            for name, param in self._params.items():
                if param.requires_grad:
                    param.grad = per_model_grads[name]
            for name in self._shared_params:
                param = self._models[0].network.get_parameter(name)
                if param.requires_grad:
                    param.grad = per_model_grads[name].sum(0)
            self._optimizer.step()
        else:
            for param in self._params.values():
                param.grad = None
            for i, m in enumerate(self._models):
                m._optimizer.zero_grad()
                for name, param in m.network.named_parameters():
                    if param.requires_grad:
                        param.grad = per_model_grads[name][i]
                m._optimizer.step()

    def _clear_member_gradients(self):
        for m in self._models:
            for param in m.network.parameters():
                param.grad = None

    def _compute_val_loss(self, val_args, use_weights, network_kwargs):
        if use_weights:
            weights = torch.as_tensor(val_args[-1], device=self._device).float()
            val_args = val_args[:-1]
        else:
            weights = None

        torch_batch = [torch.as_tensor(x, device=self._device) for x in val_args]
        n_kwargs = len(self._kwargs_ndims)
        x_end = len(torch_batch) - self._trainer.n_fit_targets - n_kwargs
        x_inputs = tuple(torch_batch[:x_end])
        kwargs_data = dict(zip(self._kwargs_ndims, torch_batch[x_end:x_end + n_kwargs]))
        y_targets = [yi.clone().detach() for yi in torch_batch[-self._trainer.n_fit_targets:]]

        base = self._base_model
        trainer = self._trainer

        def compute_loss(params, buffers):
            y_hat = functional_call(base, (params, buffers), x_inputs, kwargs={**network_kwargs, **kwargs_data})
            return trainer.compute_loss_from_output(y_hat, y_targets, weights)

        with torch.no_grad():
            losses = vmap(compute_loss, randomness=self._randomness)(*self._functional_state())
        return float(losses.mean().item())

    def _store_loss(self, losses):
        for m, ml in zip(self._models, losses):
            m._last_loss = ml

    def _log(self):
        for m in self._models:
            m._log()

    @staticmethod
    def _meta_template(network):
        memo = {id(p): torch.nn.Parameter(p.detach().to('meta'), requires_grad=p.requires_grad)
                for p in network.parameters()}
        memo.update({id(b): b.to('meta') for b in network.buffers()})

        return deepcopy(network, memo)

    @staticmethod
    def _find_shared_objects(networks, tensor_names):
        modules = [name for name, module in networks[0].named_modules()
                   if name and all(network.get_submodule(name) is module for network in networks)]
        names = [name for name in modules + tensor_names
                 if not any(name.startswith(module + '.') for module in modules)]

        return {name: TorchEnsemble._get_object(networks[0], name) for name in names}

    @staticmethod
    def _get_object(network, name):
        module_name, _, attribute = name.rpartition('.')

        return getattr(network.get_submodule(module_name), attribute)

    @staticmethod
    def _shared_names(networks, getter, names):
        shared = list()
        for name in names:
            tensors = [getattr(network, getter)(name) for network in networks]
            if all(t is tensors[0] for t in tensors):
                shared.append(name)
            else:
                assert len({id(t) for t in tensors}) == len(tensors), \
                    f'{name} is shared by some models of the ensemble but not by all of them.'

        return shared
