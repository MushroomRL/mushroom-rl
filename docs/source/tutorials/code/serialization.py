from mushroom_rl.core import Serializable

import torch
import numpy as np
from mushroom_rl.rl_utils.parameters import Parameter


class TestClass(Serializable):
    def __init__(self, value):
        # Create some different types of variables

        self._primitive_variable = value  # Primitive python variable
        self._numpy_vector = np.array([1, 2, 3]*value)  # Numpy array
        self._dictionary = dict(some='random', keywords=2, fill='the dictionary')  # A dictionary

        # Building a torch object
        data_array = np.ones(3)*value
        data_tensor = torch.from_numpy(data_array)
        self._torch_object = torch.nn.Parameter(data_tensor)

        # Some variables that implement the Serializable interface
        self._mushroom_parameter = Parameter(2.0*value)
        self._list_of_objects = [Parameter(i) for i in range(value)]  # This is a list!

        # A variable that is not important e.g. a buffer
        self.not_important = np.zeros(10000)

        # A variable that contains a reference to another variable
        self._list_reference = [self._dictionary]

        # Superclass constructor
        super().__init__()

        # Here we specify how to save each component
        self._add_save_attr(
            _primitive_variable='primitive',
            _numpy_vector='numpy',
            _dictionary='pickle',
            _torch_object='torch',
            _mushroom_parameter='mushroom',
            # List of mushroom objects can also be saved with the 'mushroom' mode
            _list_of_objects='mushroom',
            # The '!' is to specify that we save the variable only if full_save is True
            not_important='numpy!',
        )

    def _post_load(self):
        if self.not_important is None:
            self.not_important = np.zeros(10000)

        self._list_reference = [self._dictionary]


def print_variables(obj):
    for label, var in vars(obj).items():
        if label != '_save_attributes':
            if isinstance(var, Parameter):
                print(f'{label}: Parameter({var()})')
            elif isinstance(var, list) and isinstance(var[0], Parameter):
                new_list = [f'Parameter({item()})' for item in var]
                print(f'{label}:  {new_list}')
            else:
                print(label, ': ', var)


if __name__ == '__main__':
    # Create test object and print its variables
    test_object = TestClass(1)
    print('###########################################################################################################')
    print('The test object contains the following:')
    print('-----------------------------------------------------------------------------------------------------------')
    print_variables(test_object)

    # Changing the buffer
    test_object.not_important[0] = 1

    # Save the object on disk
    test_object.save('test.msh')

    # Create another test object
    test_object = TestClass(2)
    print('###########################################################################################################')
    print('After overwriting the test object:')
    print('-----------------------------------------------------------------------------------------------------------')
    print_variables(test_object)

    # Changing the buffer again
    test_object.not_important[0] = 1

    # Save the other test object, this time remember buffer
    test_object.save('test_full.msh', full_save=True)

    # Load first test object and print its variables
    print('###########################################################################################################')
    test_object = TestClass.load('test.msh')
    print('Loading previous test object:')
    print('-----------------------------------------------------------------------------------------------------------')
    print_variables(test_object)

    # Load second test object and print its variables
    print('###########################################################################################################')
    test_object = TestClass.load('test_full.msh')
    print('Loading previous test object:')
    print('-----------------------------------------------------------------------------------------------------------')
    print_variables(test_object)
