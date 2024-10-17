from collections import UserDict
import numbers
from .array_backend import ArrayBackend
from mushroom_rl.core.serialization import Serializable

class ExtraInfo(Serializable, UserDict):
    """
    A class to to collect and parse step information
    """
    def __init__(self, n_envs, backend, device=None):
        """
        Constructor.

        Args:
            n_envs (int): Number of parallel environments
        """
        self._n_envs = n_envs
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._device = device

        self._storage = []
        self._key_mapping = {} #maps keys for future output to key paths
        self._shape_mapping = {} #maps keys to additional shapes for arrays
        self._structured_storage = {}
        super().__init__()
        self._add_all_save_attr()

    def append(self, info):
        """
        Append new step information

        Args:
            info (dict or list): Information to append either list of dicts of every environment, or a dictionary of arrays 
        """
        if self._n_envs > 1:
            assert isinstance(info, (dict, list))
        else:
            assert isinstance(info, dict)

        self._storage.append(info)

    def parse(self, to=None):
        """
        Parse the stored information into an flat dictionary of arrays

        Args:
            to (str): the backend to be used for the returned arrays, 'torch' or 'numpy'.

        Returns:
            dict: Flat dictionary containing an array for every property of the step information
        """
        if to is None:
            to = self._array_backend.get_backend_name()

        #create key mapping
        for step_data in self._storage:
            if isinstance(step_data, dict):
                self._update_key_mapping(step_data, self._n_envs == 1)
            elif isinstance(step_data, list):
                for env_data in step_data:
                    assert isinstance(env_data, dict)
                    self._update_key_mapping(env_data, True)

        # calculate the size for the array
        if self._structured_storage:
            length_structured_storage = self._structured_storage[next(iter(self._structured_storage.keys()))].shape[0]
        else:
            length_structured_storage = 0
        size = (len(self._storage) + length_structured_storage, self._n_envs) if self._n_envs > 1 else (len(self._storage) + length_structured_storage, )
        
        #create output dictionary with empty arrays
        output = {
            key: ArrayBackend.get_array_backend(to).empty(size + self._shape_mapping[key], self._device)
            for key in self._key_mapping
        }

        #fill output with elements stored in structured storage
        if self._structured_storage:
            for key in output:
                index = length_structured_storage
                value = self._convert(self._structured_storage[key], to)
                output[key][:index] = value
        
        #fill output with elements stored in storage
        for index, step_data in enumerate(self._storage): 
            index = index + length_structured_storage
            if isinstance(step_data, dict):
                self._append_dict_to_output(output, step_data, index, to)
            elif isinstance(step_data, list):
                self._append_list_to_output(output, step_data, index, to)

        self._structured_storage = {key: value for key, value in output.items()}
        self._storage = []
        self._array_backend = ArrayBackend.get_array_backend(to)
            
        self.data = output
    
    def flatten(self, mask=None):
        """
        Flattens the arrays in data by combining the first two dimensions.

        Args:
            mask
        
        Returns:
            ExtraInfo: Flattened ExtraInfo
        """
        self.parse()

        info = ExtraInfo(1, self._array_backend.get_backend_name(), self._device)
        info._shape_mapping = self._shape_mapping
        info._key_mapping = self._key_mapping
        info._structured_storage = {}

        for key in self.data:
            if mask is None:
                info.data[key] = info._array_backend.flatten(self.data[key])
            else:
                info.data[key] = info._array_backend.pack_padded_sequence(self.data[key], mask)

        for key in self._structured_storage:
            if mask is None:
                info._structured_storage[key] = info._array_backend.flatten(self._structured_storage[key])
            else:
                info._structured_storage[key] = info._array_backend.pack_padded_sequence(self._structured_storage[key], mask)

        return info
        
    def __add__(self, other):
        """
        Returns new object which combines two ExtraInfo objects.

        Args:
            other(ExtraInfo): other ExtraInfo which will be combined with self
        """
        assert(self._n_envs == other.n_envs)

        info = ExtraInfo(self._n_envs, self._array_backend.get_backend_name(), self._device)
        info._storage = self._storage + other._storage

        info._structured_storage = self._concatenate_dictionary(self._structured_storage, other._structured_storage, self._array_backend, other._array_backend)
        info.data = self._concatenate_dictionary(self.data, other.data, self._array_backend, other._array_backend)

        #combine key_mapping
        info._key_mapping = self._key_mapping.copy()
        info._key_mapping.update(other._key_mapping)

        #combine shape_mapping
        info._shape_mapping = self._shape_mapping.copy()
        info._shape_mapping.update(other._shape_mapping)
        
        return info
    
    def _concatenate_array(self, array1, array2, intended_length_array1, intended_length_array2, array1_backend, array2_backend):
        """
        Concatenate array1 with array2

        Args:
            array1 (array, None)
            array2 (array, None)
            intended_length_array1 (int): Intended Length of array1 in case array1 is None
            intended_length_array2 (int): Intended Length of array2 in case array2 is None
            array1_backend (ArrayBackend): Backend of array1
            array2_backend (ArrayBackend): Backend of array2
        
        Returns:
            array: Concatenation of array1 and array2
        """
        if array1 is None:
            shape = (intended_length_array1,) + array2_backend.shape(array2)[1:]
            array1 = array1_backend.full(shape, array1_backend.none())
        if array2 is None:
            shape = (intended_length_array2, ) +  array1_backend.shape(array1)[1:]
            array2 = array2_backend.full(shape, array2_backend.none())
        array2 = array1_backend.convert(array2, backend=array2_backend)
        return array1_backend.concatenate((array1, array2))
    
    def _concatenate_dictionary(self, dict1, dict2, backend1, backend2):
        """
        Concatenate dict1 with dict2.

        Args:
            dict1 (dict): Flat dictionary containing arrays of backend1
            dict2 (dict): Flat dictionary containing arrays of backend2
            backend1 (ArrayBackend): Backend of arrays in dict1.
            backend2 (ArrayBackend): Backend of arrays in dict2.

        Returns
            dict: Concatenation of dict1 and dict2
        """
        if not dict1:
            return dict2
        if not dict2:
            return dict1
        
        array_length_dict1 = backend1.shape(dict1[next(iter(dict1.keys()))])[0]
        array_length_dict2 = backend2.shape(dict2[next(iter(dict2.keys()))])[0]

        r = {}

        for key in dict1.keys() | dict2.keys():
            array1 = dict1[key] if key in dict1 else None
            array2 = dict2[key] if key in dict2 else None
            r[key] = self._concatenate_array(array1, array2, array_length_dict1, array_length_dict2, backend1, backend2)
        return r


    def copy(self):
        info = ExtraInfo(self._n_envs, self._array_backend.get_backend_name(), self._device)
        info._storage = self._storage.copy()
        info._key_mapping = self._key_mapping.copy()
        info._shape_mapping = self._shape_mapping.copy()
        info.data = self.data.copy()
            
        return info

    def get_view(self, index, copy=False):
        """
        Returns ExtraInfo Object which only contains the specified indexes

        Args:
            index (int, slice, ndarray, tensor): indexes which the return should contain
            copy (bool): wether content of ExtraInfo object should be copied
        """
        self.parse()
        info = ExtraInfo(self._n_envs, self._array_backend.get_backend_name(), self._device)
        info._key_mapping = self._key_mapping
        info._shape_mapping = self._shape_mapping

        if not copy:
            info._structured_storage = {key: value[index, ...] for key, value in self._structured_storage.items()}
            info.data = {key: value[index, ...] for key, value in self.data.items()}
        else:
            for key, value in self._structured_storage.items():
                value = value[index, ...]
                info._structured_storage[key] = self._array_backend.empty(value.shape, self._device)
                info._structured_storage[key][:] = value
            
            for key, value in self.data.items():
                value = value[index, ...]
                info.data[key] = self._array_backend.empty(value.shape, self._device)
                info.data[key][:] = value
        
        return info
    
    def clear(self):
        self._storage = []
        self._key_mapping = {}
        self._shape_mapping = {}
        self._structured_storage = {}
        self.data = {}

    def _add_all_save_attr(self):
        self._add_save_attr(
            data='primitive',
            _storage='primitive',
            _structured_storage='primitive',
            _key_mapping='primitive',
            _shape_mapping='primitive'
        )

    def _update_key_mapping(self, template, single_env):
        """
        Update the pattern and the key_paths with the keys from the given template

        Args:
            template (dict): Dictionary to extract the keys from
            single_env (bool): Wether template contains data for only one environment
        """
        assert(isinstance(template, dict))

        # Stack to store dictionaries and their parent key
        stack = [(template, [])]

        while stack:
            structure_element, parent_keys = stack.pop()
            assert isinstance(structure_element, dict)

            #Iterate over the dict
            for key, value in structure_element.items():
                key_path = parent_keys + [key]

                # skip if key is already in key_mapping
                if key_path in self._key_mapping.values():
                    continue

                if isinstance(value, dict):
                    stack.append((value, key_path))
                else:
                    new_key = self._create_key(key_path)
                    self._store_array_shape(new_key, value, single_env)

    def _append_dict_to_output(self, output, step_data, index, to):
        """
        Append a dictionary to the output arrays.

        Args:
            output (dict): Flat dictionary containing the arrays
            step_data (dict): Containing the step information for one step
            index (int): index of the step
            to (str): Target format
        """
        for key, key_path in self._key_mapping.items():
            value = self._find_element_by_key_path(step_data, key_path)
            value = self._convert(value, to)
            output[key][index] = value
    
    def _append_list_to_output(self, output, step_data, index, to):
        """
        Append a list to the output arrays.

        Args:
            output (list): Flat dictionary containing the arrays
            step_data (dict): List containing the step information in form of a dictionary for every environment
            index (int): index of the step
            to (str): Target format
        """
        assert(self._n_envs > 1)
        for key, key_path in self._key_mapping.items():
            for i, env_data in enumerate(step_data):
                value = self._find_element_by_key_path(env_data, key_path)
                value = self._convert(value, to)
                output[key][index][i] = value

    def _find_element_by_key_path(self, source, key_path):
        """
        Find the value in source corresponding to the key path.

        Args:
            source (dict): Dictionary to search in.
            key_path (list): List of keys.

        Returns:
            The found value or None if any key is missing.
        """
        current = source
        for key in key_path:
            if key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _convert(self, value, to):
        """
        Convert value to the target format.

        Args:
            value: Value to convert.
            to (str): Target format, 'torch' or 'numpy'.

        Returns:
            Converted value.
        """
        if isinstance(value, numbers.Number):
            return value
        
        if value is None:
            return ArrayBackend.get_array_backend(to).none()
        
        return ArrayBackend.convert(value, to=to, backend=self._array_backend)

    def _create_key(self, key_path):
        """
        Creates single key in pattern from a list of keys.

        Args:
            key_path (list): List of keys to combine.
        
        Returns:
            key (str): Created key.
        """
        key = "_".join(str(key) for key in key_path)
        self._key_mapping[key] = key_path
        return key

    def _store_array_shape(self, key, value, single_env):
        """
        Stores the shape of the value. If value does not have a shape, an empty tuple is stored.

        Args:
            key (str): Dictionary key.
            value (Array, Number): Variable whose shape should be saved
            sinlge_env (bool): 
        """
        if isinstance(value, numbers.Number):
            self._shape_mapping[key] = ()
        else:
            shape = self._array_backend.shape(value)
            self._shape_mapping[key] = shape[1:] if not single_env else shape
    
    @property
    def n_envs(self):
        return self._n_envs

    def __setitem__(self, key, value):
        raise TypeError("This dictionary is read-only.")

    def __delitem__(self, key):
        raise TypeError("This dictionary is read-only.")

    def pop(self, key, default=None):
        raise TypeError("This dictionary is read-only.")

    def popitem(self):
        raise TypeError("This dictionary is read-only.")

    def setdefault(self, key, default=None):
        raise TypeError("This dictionary is read-only.")

    def update(self, *args, **kwargs):
        raise TypeError("This dictionary is read-only.")