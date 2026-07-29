import torch

from mushroom_rl.core.mushroom_object import MushroomObject
from mushroom_rl.approximators.approximator import Ensemble
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator


class RecurrentTorchApproximator(TorchApproximator):
    """
    Class extending the TorchApproximator for the recurrent-network setting. Its ``predict`` shapes the online inputs
    into the ``(batch, sequence, feature)`` layout the recurrent network expects: it batch-pads a single sample, inserts
    a length-1 sequence dimension for the online case, handles the ``policy_state`` (batch dimension only), and
    defaults ``lengths`` to ones. Training tensors already carry both the batch and sequence dimensions and pass
    through unchanged. When ``n_models > 1`` construction dispatches to ``RecurrentTorchEnsemble``.

    """

    def __new__(cls, *args, n_models=None, **kwargs):
        if cls is RecurrentTorchApproximator and n_models is not None and n_models > 1:
            instance = MushroomObject.__new__(RecurrentTorchEnsemble)
            RecurrentTorchEnsemble.__init__(instance, *args, n_models=n_models, **kwargs)
            return instance
        else:
            return MushroomObject.__new__(cls)

    def __init__(self, network, input_shape, output_shape, policy_state_shape, optimizer=None, loss=None,
                 batch_size=0, n_fit_targets=1, reinitialize=False, dropout=False, quiet=True, n_models=None,
                 action_history_shape=None, **params):
        """
        Constructor.

        Args:
            network (torch.nn.Module): the recurrent network class to use;
            input_shape (tuple): the per-timestep shape of the observation;
            output_shape (tuple, list): shape of the output of the network. A list of shape tuples for a
                network returning several tensors (e.g. an actor returning ``[action, next_policy_state]``);
            policy_state_shape (tuple): the shape of the recurrent hidden state carried as policy state;
            action_history_shape (tuple, None): the per-timestep shape of the previous-action input, when the
                network consumes the action history;
            **params: parameters forwarded to :class:`TorchApproximator` and the network constructor.

        """
        super().__init__(network, input_shape, output_shape, optimizer=optimizer, loss=loss,
                         batch_size=batch_size, n_fit_targets=n_fit_targets, reinitialize=reinitialize,
                         dropout=dropout, quiet=quiet, action_history_shape=action_history_shape, **params)

        self._policy_state_ndim = len(policy_state_shape)

        self._add_save_attr(_policy_state_ndim='primitive')

    def predict(self, state, policy_state, lengths=None, **kwargs):
        """
        Predict. The observation and every declared keyword input (e.g. ``action_history``) are shaped into the
        ``(batch, sequence, feature)`` layout: a single online sample gets both a batch and a length-1 sequence
        dimension, a batch of online samples gets only the sequence dimension, and full training sequences pass
        through unchanged. A declared keyword input that is not passed raises a ``KeyError``.

        Args:
            state (torch.Tensor): the observation or observation sequence;
            policy_state (torch.Tensor): the recurrent hidden state;
            lengths (torch.Tensor, None): the length of each input sequence, defaulted to ones when None;
            **kwargs: additional per-timestep keyword inputs (e.g. ``action_history``).

        Returns:
            The predictions of the model.

        """
        state = self._pad_sequence(state, self._input_ndims[0])
        policy_state = self._pad_batch(policy_state, self._policy_state_ndim)
        for name, ndim in self._kwargs_ndims.items():
            kwargs[name] = self._pad_sequence(kwargs[name], ndim)

        if lengths is None:
            lengths = torch.ones(state.shape[0], dtype=torch.long)

        return self._parse_output(self.network(state, policy_state, lengths, **kwargs))

    def diff(self, *args, **kwargs):
        """
        Not supported: ``diff`` follows the feedforward calling convention, which does not carry the sequence
        lengths a recurrent network needs.

        """
        raise NotImplementedError('diff is not supported by recurrent approximators.')

    @staticmethod
    def _pad_sequence(x, ndim):
        if x.ndim == ndim:
            x = x.unsqueeze(0)
        if x.ndim == ndim + 1:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _pad_batch(x, ndim):
        return x.unsqueeze(0) if x.ndim == ndim else x


class RecurrentTorchEnsemble(Ensemble):
    """
    Ensemble of RecurrentTorchApproximator models. Unlike ``TorchEnsemble`` it aggregates the models with the
    base ``Ensemble`` loop rather than ``torch.func.vmap``, since the recurrent ``pack_padded_sequence`` does not
    compose with ``vmap``. Constructed automatically by ``RecurrentTorchApproximator`` when ``n_models > 1``.

    """

    def __init__(self, network, input_shape, output_shape, policy_state_shape, optimizer=None, loss=None,
                 batch_size=0, n_fit_targets=1, reinitialize=False, dropout=False, quiet=True, n_models=None,
                 prediction='mean', action_history_shape=None, **params):
        """
        Constructor.

        Args:
            n_models (int): number of models in the ensemble;
            prediction (str, 'mean'): how to aggregate predictions across models. One of ``'mean'``, ``'min'``,
                ``'max'``, ``'sum'``, or ``'all'`` to return all predictions;
            **params: parameters forwarded to each :class:`RecurrentTorchApproximator`.

        """
        super().__init__(RecurrentTorchApproximator, n_models, prediction=prediction, backend='torch',
                         input_shape=input_shape, output_shape=output_shape, network=network,
                         policy_state_shape=policy_state_shape, optimizer=optimizer, loss=loss,
                         batch_size=batch_size, n_fit_targets=n_fit_targets, reinitialize=reinitialize,
                         dropout=dropout, quiet=quiet, action_history_shape=action_history_shape, **params)

    def parameters(self):
        """
        Returns:
            The concatenated parameters of all models in the ensemble.

        """
        return [p for m in self._models for p in m.parameters()]

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

    def get_weights(self):
        """
        Returns:
            The stacked weights of all models, shape ``(n_models, weights_size_per_model)``.

        """
        return torch.stack([m.get_weights() for m in self._models])

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. the input for each model, stacked.

        Args:
            *args: input;
            **kwargs: other parameters used by the diff method of the regressor.

        Returns:
            The stacked derivatives of all models w.r.t. the input.

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
