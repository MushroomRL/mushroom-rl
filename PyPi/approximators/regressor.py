from sklearn import preprocessing


class Regressor(object):
    def __init__(self, approximator_class, **apprx_params):
        self.features = apprx_params.pop('features', None)
        self.input_scaled = apprx_params.pop('input_scaled', True)
        self.output_scaled = apprx_params.pop('output_scaled', True)

        self.model = approximator_class(**apprx_params)

    def fit(self, x, y, **fit_params):
        if self.features:
            x = self.features.fit_transform(x)

        if self.input_scaled:
            self.pre_x = preprocessing.StandardScaler()
            x = self.pre_x.fit_transform(x)

        if self.output_scaled:
            self.pre_y = preprocessing.StandardScaler()
            y = self.pre_y.fit_transform(y)

        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        if self.features:
            x = self.features.transform(x)

        if self.input_scaled:
            x = self.pre_x.transform(x)

        y = self.model.predict(x)

        return self.pre_y.inverse_transform(y) if self.output_scaled else y
