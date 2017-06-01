import numpy as np

from PyPi.approximators.action_regressor import ActionRegressor
from PyPi.approximators.regressor import Regressor


class Ensemble(object):
    """
    This class implements functions to manage regressor ensembles.
    """
    def __init__(self, approximator_class, n_models, discrete_actions=None,
                 **params):
        """
        Constructor.

        # Arguments
            approximator_class (object): the model class to approximate the
            Q-function of each action.
            discrete_actions (np.array): the values of the discrete actions.
            **params (dict): parameters dictionary to construct each regressor.
        """
        self.n_models = n_models
        self.fit_actions = True if discrete_actions is None else True
        self.models = list()

        if not self.fit_actions:
            regressor_class = ActionRegressor
            params['discrete_actions'] = discrete_actions
        else:
            regressor_class = Regressor

        for _ in range(self.n_models):
            self.models.append(regressor_class(approximator_class, **params))

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states and actions.

        # Returns
            The predictions of the model.
        """
        predictions = np.zeros((x[0].shape[0]))
        for i in range(self.n_models):
            predictions += self.models[i].predict([x[0], x[1]])

        predictions /= self.n_models

        return predictions

    def __getitem__(self, idx):
        return self.models[idx]

    def __str__(self):
        s = '.' if self.fit_actions else ' with action regression.'
        return 'Ensemble of %d ' % self.n_models + str(self.models[0]) + s
