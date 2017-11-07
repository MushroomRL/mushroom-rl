import numpy as np
from mushroom.policy import GaussianPolicy, MultivariateGaussianPolicy
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.parameters import Parameter

input_shape = (2,)
approximator_params = dict(params=np.array([1.0, 0.5]))
approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=(1,),
                             params=approximator_params)

sigma_p = Parameter(value=.05)
policy_1 = GaussianPolicy(mu=approximator, sigma=sigma_p)

sigma = np.array([[.05**2]])
policy_2 = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)


state = np.array([1.3, 2.4])
action = np.array([2.4])

print 'state: ', state
print 'action: ', action

print 'Standard gaussian'
print 'p(action|state): ', policy_1(state, action)
print 'd log(p(action|state)): ', policy_1.diff_log(state, action)

print 'Multivariate gaussian, single variable'
print 'p(action|state): ', policy_2(state, action)
print 'd log(p(action|state)): ', policy_2.diff_log(state, action)


approximator_params_m = dict(params=np.array([1.0, 0.0, 0.0, 1.0]), output_dim=2)
approximator_m = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=(2,),
                             params=approximator_params_m)

sigma_m = np.eye(2, 2)
policy_m = MultivariateGaussianPolicy(mu=approximator_m, sigma=sigma_m)

state = np.array([-1.2731,  -2.4746])
action = np.array([-2.2535,  -2.7151])

print 'state: ', state
print 'action: ', action

print 'Multivariate gaussian, multiple variable'
print 'p(action|state): ', policy_m(state, action)
print 'd p(action|state): ', policy_m.diff(state, action)
