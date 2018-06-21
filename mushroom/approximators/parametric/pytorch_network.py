import torch
from tqdm import trange, tqdm

from mushroom.utils.minibatches import minibatch_generator


class PyTorchApproximator:
    """
    Class to interface a pytorch model to the mushroom Regressor interface.
    This class implements all is needed to use a generic pytorch model and train
    it using a specified optimizer and objective function.
    This class supports also minibatches.

    """
    def __init__(self, network, optimizer, loss, n_epochs=1, batch_size=0,
                 quiet=True, **params):
        """
        Constructor.

        Args:
            network (torch.nn.Module): the network class to use;
            optimizer (torch.optim): the optimizer used for every fit step;
            loss (torch.nn.functional): the loss function to optimize in the
                fit method;
            n_epochs (int, 1): the number of epochs to run during the fit;
            batch_size (int, 0): the size of each minibatch. If 0, the whole
                dataset is fed to the optimizer at each epoch;
            quiet (bool, True): if False, shows two progress bars, one for
                epochs and one for the minibatches;
            params (dict): dictionary of parameters needed to construct the
                network.

        """
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._quiet = quiet

        self._network = network(params)
        self._optimizer = optimizer['class'](self._network.parameters(),
                                             **optimizer['params'])
        self._loss = loss

    def predict(self, s):
        s = torch.from_numpy(s)
        val = self._network.forward(s).detach().numpy()

        return val

    def fit(self, *args):
        if self._batch_size > 0:
            batches = minibatch_generator(self._batch_size, args)
        else:
            batches = [args]

        for _ in trange(self._n_epochs, disable=self._quiet):
            for batch in tqdm(batches, disable=self._quiet):
                if len(args) == 3:
                    s = torch.from_numpy(batch[0])
                    a = torch.from_numpy(batch[1]).long()
                    q = torch.from_numpy(batch[2])

                    x = [s, a]
                    y = q

                elif len(args) == 2:
                    x = [torch.from_numpy(batch[0])]
                    y = torch.from_numpy(batch[1])

                y_hat = self._network(*x)
                loss = self._loss(y_hat, y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

    def set_weights(self, weights):
        self._network.load_state_dict(weights)

    def get_weights(self):
        return self._network.state_dict()
