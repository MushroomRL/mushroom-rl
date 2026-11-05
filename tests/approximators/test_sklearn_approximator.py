import numpy as np
import pytest
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge

from mushroom_rl.core import MushroomObject
from mushroom_rl.approximators import Ensemble, QApproximator
from mushroom_rl.approximators.sklearn_approximator import SklearnApproximator


def test_sklearn_approximator():
    np.random.seed(1)
    x = np.random.rand(20, 3)
    y = x @ np.array([1., -2., .5]) + .3

    approximator = SklearnApproximator(LinearRegression, input_shape=(3,))

    assert approximator.input_shape == (3,)
    assert approximator.output_shape == (1,)
    assert approximator.backend_name == 'numpy'
    assert approximator.device is None
    assert isinstance(approximator.model, LinearRegression)

    approximator.fit(x, y)
    y_hat = approximator.predict(np.array([[.1, .2, .3], [.4, .5, .6]]))

    assert y_hat.shape == (2,)
    assert np.allclose(y_hat, np.array([.15, 0.]))
    assert np.array_equal(approximator(x[:2]), approximator.predict(x[:2]))


def test_sklearn_approximator_matches_estimator():
    np.random.seed(1)
    x = np.random.rand(50, 2)
    y = np.random.rand(50)
    x_test = np.random.rand(10, 2)

    approximator = SklearnApproximator(ExtraTreesRegressor, input_shape=(2,), n_estimators=10, random_state=3)
    estimator = ExtraTreesRegressor(n_estimators=10, random_state=3)

    assert approximator.model.get_params() == estimator.get_params()

    approximator.fit(x, y)
    estimator.fit(x, y)

    assert np.array_equal(approximator.predict(x_test), estimator.predict(x_test))


def test_sklearn_approximator_fit_and_predict_params():
    np.random.seed(1)
    x = np.random.rand(20, 2)
    y = np.random.rand(20)
    weights = np.random.rand(20)

    approximator = SklearnApproximator(Ridge, input_shape=(2,), alpha=.5)
    estimator = Ridge(alpha=.5)

    approximator.fit(x, y, sample_weight=weights)
    estimator.fit(x, y, sample_weight=weights)

    assert np.array_equal(approximator.predict(x), estimator.predict(x))

    approximator = SklearnApproximator(ExtraTreesRegressor, input_shape=(2,), n_estimators=5, random_state=0)
    approximator.fit(x, y)

    with pytest.raises(TypeError):
        approximator.predict(x, unknown_param=True)

    approximator = SklearnApproximator(BayesianRidge, input_shape=(2,), output_shape=[(1,), (1,)])
    estimator = BayesianRidge()

    approximator.fit(x, y)
    estimator.fit(x, y)
    mean, std = approximator.predict(x, return_std=True)
    mean_estimator, std_estimator = estimator.predict(x, return_std=True)

    assert np.array_equal(mean, mean_estimator) and np.array_equal(std, std_estimator)
    assert np.array_equal(approximator.predict(x), mean_estimator)
    assert np.array_equal(approximator.predict(x[0]), mean_estimator[:1])


def test_sklearn_approximator_forwards_estimator_param():
    np.random.seed(1)
    x = np.random.rand(20, 2)
    y = np.random.rand(20)

    approximator = SklearnApproximator(BaggingRegressor, input_shape=(2,), estimator=Ridge(alpha=.5), n_estimators=3,
                                       random_state=0)
    estimator = BaggingRegressor(estimator=Ridge(alpha=.5), n_estimators=3, random_state=0)

    assert isinstance(approximator.model.estimator, Ridge)

    approximator.fit(x, y)
    estimator.fit(x, y)

    assert np.array_equal(approximator.predict(x), estimator.predict(x))


def test_sklearn_approximator_not_fitted():
    approximator = SklearnApproximator(LinearRegression, input_shape=(2,))

    with pytest.raises(NotFittedError):
        approximator.predict(np.zeros((1, 2)))


def test_sklearn_approximator_requires_input_shape():
    with pytest.raises(TypeError):
        SklearnApproximator(LinearRegression)


def test_sklearn_approximator_ensemble():
    np.random.seed(1)
    x = np.random.rand(30, 2)
    y = np.random.rand(30)

    ensemble = SklearnApproximator(model_class=ExtraTreesRegressor, input_shape=(2,), n_models=3, n_estimators=5)

    assert isinstance(ensemble, Ensemble)
    assert len(ensemble) == 3
    assert ensemble.backend_name == 'numpy'
    assert all(isinstance(m, SklearnApproximator) for m in ensemble)

    with pytest.raises(NotFittedError):
        ensemble.predict(x, idx=0)

    ensemble.fit(x, y)
    predictions = ensemble.predict(x, prediction='all')

    assert predictions.shape == (3, 30)
    assert np.allclose(ensemble.predict(x), predictions.mean(0))


def test_sklearn_q_approximator():
    np.random.seed(1)
    n_actions = 2
    s = np.random.rand(100, 2)
    a = np.random.randint(n_actions, size=(100, 1))
    q = np.random.rand(100)
    s_test = np.random.rand(4, 2)
    a_test = np.array([[0], [1], [1], [0]])

    approximator = QApproximator(SklearnApproximator, n_actions=n_actions, input_shape=(2,),
                                 model_class=LinearRegression)
    approximator.fit(s, a, q)

    estimators = [LinearRegression().fit(s[a[:, 0] == i], q[a[:, 0] == i]) for i in range(n_actions)]
    q_test = np.stack([e.predict(s_test) for e in estimators], axis=1)

    assert all(isinstance(m, SklearnApproximator) for m in approximator.model)
    assert np.allclose(approximator.predict(s_test), q_test)
    assert np.allclose(approximator.predict(s_test, a_test), q_test[np.arange(4), a_test[:, 0]])


def test_q_approximator_rejects_raw_estimator():
    with pytest.raises(TypeError):
        QApproximator(ExtraTreesRegressor, n_actions=2, input_shape=(2,))


def test_sklearn_approximator_save_load(tmpdir):
    np.random.seed(1)
    x = np.random.rand(30, 2)
    y = np.random.rand(30)

    approximator = SklearnApproximator(ExtraTreesRegressor, input_shape=(2,), n_estimators=5)
    ensemble = SklearnApproximator(model_class=ExtraTreesRegressor, input_shape=(2,), n_models=2, n_estimators=5)
    approximator.fit(x, y)
    ensemble.fit(x, y)

    approximator.save(tmpdir / 'approximator')
    ensemble.save(tmpdir / 'ensemble')
    approximator_load = MushroomObject.load(tmpdir / 'approximator')
    ensemble_load = MushroomObject.load(tmpdir / 'ensemble')

    assert approximator_load.input_shape == (2,)
    assert approximator_load.output_shape == (1,)
    assert approximator_load.backend_name == 'numpy'
    assert np.array_equal(approximator_load.predict(x), approximator.predict(x))
    assert np.array_equal(ensemble_load.predict(x, prediction='all'), ensemble.predict(x, prediction='all'))


def test_sklearn_approximator_shapes():
    x = [[0., 1.], [1., 0.], [1., 1.], [2., 1.]]
    multi = SklearnApproximator(LinearRegression, (2,), output_shape=(2,))
    multi.fit(x, [[1., 2.], [2., 1.], [3., 3.], [4., 3.]])

    assert multi.predict(x).shape == (4, 2)
    assert multi.predict(x[0]).shape == (2,)

    single = SklearnApproximator(LinearRegression, (2,))
    single.fit(x, [1., 2., 3., 4.])

    assert single.predict(x).shape == (4,) and single.predict(x[0]).shape == (1,)

    with pytest.raises(AssertionError):
        single.predict(np.arange(4.))
    with pytest.raises(AssertionError):
        single.fit(np.zeros((4, 3)), np.zeros(4))

    std = SklearnApproximator(BayesianRidge, (2,), output_shape=[(1,), (1,)])
    std.fit(x, [1., 2., 3., 4.])
    mean, deviation = std.predict(x, return_std=True)
    mean_single, deviation_single = std.predict(x[0], return_std=True)

    assert mean.shape == (4,) and deviation.shape == (4,) and np.all(deviation > 0)
    assert mean_single.shape == (1,) and deviation_single.shape == (1,)
    assert np.allclose(mean_single, mean[:1]) and np.allclose(deviation_single, deviation[:1])
