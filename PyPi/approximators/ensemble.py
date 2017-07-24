import numpy as np

from PyPi.approximators.action_regressor import ActionRegressor
from PyPi.approximators.regressor import Regressor


class Ensemble(object):
    """
    This class implements functions to manage regressor ensembles.
    """
    def __init__(self, approximator, n_models, action_space=None,
                 **params):
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
        self.fit_actions = True if action_space is None else True
        self.models = list()

        if not self.fit_actions:
            regressor_class = ActionRegressor
            params['action_space'] = action_space
        else:
            regressor_class = Regressor

        for _ in xrange(self.n_models):
            self.models.append(regressor_class(approximator, **params))

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states and actions.

        # Returns
            The predictions of the model.
        """
        y = np.zeros((x[0].shape[0]))
        for i in xrange(self.n_models):
            y += self.models[i].predict(x)

        y /= self.n_models

        return y

    def predict_all(self, x, actions):
        """
        Predict Q-value for each action given a state.

        # Arguments
            x (np.array): input dataset containing states;
            actions (np.array): list of actions of the MDP.

        # Returns
            The predictions of the model.
        """
        y = np.zeros((x.shape[0], actions.shape[0]))
        for i in xrange(self.n_models):
            y += self.models[i].predict_all(x, actions)

        y /= self.n_models

        return y

    def __getitem__(self, idx):
        return self.models[idx]

    def __str__(self):
        s = '.' if self.fit_actions else ' with action regression.'
        return 'Ensemble of %d ' % self.n_models + str(self.models[0]) + s
