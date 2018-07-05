import numpy as np

from mushroom.approximators.regressor import Regressor
from mushroom.approximators.parametric import *


def test_linear_approximator():
    np.random.seed(88)

    noise = 1e-3

    a = np.random.rand(1000, 3)

    k = np.random.rand(3, 2)
    b = a.dot(k) + np.random.randn(1000, 2)*noise

    approximator = Regressor(LinearApproximator,
                             input_shape=(3,),
                             output_shape=(2,))

    approximator.fit(a, b)

    khat = approximator.get_weights()

    deltaK = (khat - k.T.flatten())

    assert np.linalg.norm(deltaK) < noise

    point = np.random.randn(3,)
    derivative = approximator.diff(point)

    lp = len(point)
    for i in range(derivative.shape[1]):
        assert (derivative[i*lp:(i+1)*lp, i] == point).all()


if __name__ == '__main__':
    test_linear_approximator()