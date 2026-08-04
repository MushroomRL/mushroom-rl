import numpy as np
from tqdm import trange, tqdm

from mushroom_rl.core.mushroom_object import MushroomObject


class TorchTrainer(MushroomObject):
    def __init__(self, loss_fn, batch_size, n_fit_targets, reinitialize, dropout,
                 fit_epoch_fn, compute_val_loss_fn, store_loss_fn, quiet):
        self._loss_fn = loss_fn
        self.quiet = quiet
        self.batch_size = batch_size
        self.n_fit_targets = n_fit_targets
        self.reinitialize = reinitialize
        self.dropout = dropout
        self._fit_epoch_fn = fit_epoch_fn
        self._compute_val_loss_fn = compute_val_loss_fn
        self._store_loss_fn = store_loss_fn

        self._add_save_attr(
            _loss_fn='pickle',
            quiet='primitive',
            batch_size='primitive',
            n_fit_targets='primitive',
            reinitialize='primitive',
            dropout='primitive',
        )

    def set_callbacks(self, fit_epoch_fn, compute_val_loss_fn, store_loss_fn):
        self._fit_epoch_fn = fit_epoch_fn
        self._compute_val_loss_fn = compute_val_loss_fn
        self._store_loss_fn = store_loss_fn

    def fit(self, args, n_epochs, weights, epsilon, patience, validation_split, network_kwargs):
        check_loss, n_epochs, use_weights, train_args, val_args = self._setup(
            args, n_epochs, weights, epsilon, validation_split)
        self._run_loop(n_epochs, check_loss, epsilon, patience,
                       train_args, val_args, use_weights, network_kwargs)

    def _setup(self, args, n_epochs, weights, epsilon, validation_split):
        if epsilon is not None:
            n_epochs = np.inf if n_epochs is None else n_epochs
            check_loss = True
        else:
            n_epochs = 1 if n_epochs is None else n_epochs
            check_loss = False

        if weights is not None:
            args = args + (weights,)
            use_weights = True
        else:
            use_weights = False

        if not 0 < validation_split <= 1:
            raise ValueError

        train_len = np.ceil(len(args[0]) * validation_split).astype(int)
        train_args = [a[:train_len] for a in args]
        val_args = [a[train_len:] for a in args]

        return check_loss, n_epochs, use_weights, train_args, val_args

    def _run_loop(self, n_epochs, check_loss, epsilon, patience,
                  train_args, val_args, use_weights, network_kwargs):
        patience_count = 0
        best_loss = np.inf
        epochs_count = 0

        if check_loss:
            with tqdm(total=n_epochs if n_epochs < np.inf else None,
                      dynamic_ncols=True, disable=self.quiet, leave=False) as t_epochs:
                while patience_count < patience and epochs_count < n_epochs:
                    loss_current = self._fit_epoch_fn(train_args, use_weights, network_kwargs)

                    if len(val_args[0]):
                        loss = self._compute_val_loss_fn(val_args, use_weights, network_kwargs)
                    else:
                        loss = float(np.mean(loss_current))

                    if not self.quiet:
                        t_epochs.set_postfix(loss=loss)
                        t_epochs.update(1)

                    if best_loss - loss > epsilon:
                        patience_count = 0
                        best_loss = loss
                    else:
                        patience_count += 1

                    self._store_loss_fn(loss_current)
                    epochs_count += 1
        else:
            with trange(n_epochs, disable=self.quiet) as t_epochs:
                for _ in t_epochs:
                    loss_current = self._fit_epoch_fn(train_args, use_weights, network_kwargs)

                    if not self.quiet:
                        t_epochs.set_postfix(loss=float(np.mean(loss_current)))

                    self._store_loss_fn(loss_current)

    def compute_loss_from_output(self, y_hat, y_targets, weights=None):
        if isinstance(y_hat, tuple):
            output_type = y_hat[0].dtype
        else:
            output_type = y_hat.dtype
        y = [yi.to(output_type) for yi in y_targets]
        if weights is None:
            return self._loss_fn(y_hat, *y)
        loss = self._loss_fn(y_hat, *y, reduction='none')
        loss = loss @ weights
        return loss / weights.sum()
