import numpy as np
from matplotlib import pyplot as plt

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator


x = np.arange(10).reshape(-1, 1)

intercept = 10
noise = np.random.randn(10, 1) * 1
y = 2 * x + intercept + noise

phi = np.concatenate((np.ones(10).reshape(-1, 1), x), axis=1)

regressor = Regressor(LinearApproximator,
                      input_shape=(2,),
                      output_shape=(1,))

regressor.fit(phi, y)

print('Weights: ' + str(regressor.get_weights()))
print('Gradient: ' + str(regressor.diff(np.array([[5.]]))))

plt.scatter(x, y)
plt.plot(x, regressor.predict(phi))
plt.show()
