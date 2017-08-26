import numpy as np

from PyPi.approximators.action_regressor import ActionRegressor
from PyPi.approximators.regressor import Regressor


class Ensemble(object):
    """
    This class implements functions to manage regressor ensembles.
    """
    def __init__(self, approximator, n_models, use_action_regressor=False,
                 discrete_actions=None, **params):
        """
        Constructor.

        # Arguments
            approximator (object): the model class to approximate the
                Q-function of each action;
            n_models (int): number of models in the ensemble;
            action_space (object): action_space of the MDP;
            **params (dict): parameters dictionary to construct each regressor.
        """
        self.n_models = n_models
        self._use_action_regressor = use_action_regressor
        self.models = list()

        regressor_class =\
            ActionRegressor if self._use_action_regressor else Regressor

        for _ in xrange(self.n_models):
            self.models.append(
                regressor_class(approximator, discrete_actions=discrete_actions,
                                **params))

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states and actions.

        # Returns
            The predictions of the model.
        """
        y = list()
        for m in self.models:
            y.append(m.predict(x))
        y = np.mean(y, axis=0)

        return y

    def predict_all(self, x):
        """
        Predict Q-value for each action given a state.

        # Arguments
            x (np.array): input dataset containing states;
            actions (np.array): list of actions of the MDP.

        # Returns
            The predictions of the model.
        """
        y = list()
        for m in self.models:
            y.append(m.predict_all(x))
        y = np.mean(y, axis=0)

        return y

    def __getitem__(self, idx):
        return self.models[idx]

    def __str__(self):
        s = '.' if self._use_action_regressor else ' with action regression.'
        return 'Ensemble of %d ' % self.n_models + str(self.models[0]) + s
