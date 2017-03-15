from keras.models import Sequential
from keras.layers.core import Dense


class DenseNN(object):
    def __init__(self,
                 n_input,
                 n_output,
                 hidden_neurons,
                 init='glorot_uniform',
                 loss='mse',
                 metrics=[],
                 activation='relu',
                 optimizer='rmsprop',
                 regularizer=None):
        assert isinstance(hidden_neurons, list), 'hidden_neurons should be \
            of type list specifying the number of hidden neurons for each \
            hidden layer.'
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_neurons = hidden_neurons
        self.init = init
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.regularizer = regularizer
        self.optimizer = optimizer

    def fit(self, x, y, **fit_params):
        if not hasattr(self, 'model'):
            self.model = self._init()

        self.model.fit(x, y, **fit_params)

    def predict(self, x, **kwargs):
        predictions = self.model.predict(x, **kwargs)

        return predictions.ravel()

    def _init(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons[0],
                        input_shape=(self.n_input,),
                        activation=self.activation,
                        init=self.init,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))
        for i in range(1, len(self.hidden_neurons)):
            model.add(Dense(self.hidden_neurons[i],
                            activation=self.activation,
                            init=self.init,
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer))
        model.add(Dense(self.n_output,
                        activation='linear',
                        init=self.init,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return model
