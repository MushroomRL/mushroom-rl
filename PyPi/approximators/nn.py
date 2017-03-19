from keras.models import Sequential
from keras.layers.core import Dense


class DenseNN(object):
    """
    Wrapper class of Keras's Sequential model.
    """
    def __init__(self, **approximator_params):
        """
        Constructor.

        # Arguments
            approximator_params (dict): parameters.
        """
        self.__name__ = 'DenseNN'

        self.approximator_params = approximator_params

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).
            y (np.array): target.
            fit_params (dict): other parameters.
        """
        if not hasattr(self, 'model'):
            self.model = self._init()

        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the model.
        """
        predictions = self.model.predict(x)

        return predictions.ravel()

    def _init(self):
        pars = self.approximator_params
        n_input = pars.pop('n_input')
        hidden_neurons = pars.pop('hidden_neurons')
        n_output = pars.pop('n_output')
        loss = pars.pop('loss')
        optimizer = pars.pop('optimizer')

        model = Sequential()
        model.add(Dense(hidden_neurons[0], input_shape=(n_input,), **pars))
        for i in range(1, len(hidden_neurons)):
            model.add(Dense(hidden_neurons[i], **pars))
        model.add(Dense(n_output, activation='linear', **pars))

        model.compile(loss=loss, optimizer=optimizer, **pars)

        return model

    def __str__(self):
        return self.__name__
