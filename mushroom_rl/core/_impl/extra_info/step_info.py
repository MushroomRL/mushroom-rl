import numbers
from copy import deepcopy

from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.mushroom_object import MushroomObject


class StepInfo(MushroomObject):
    """
    Container for the step information an environment returns, turning it into a flat dictionary of arrays on
    demand.

    """
    def __init__(self, n_envs, backend, device=None, vectorized=None):
        """
        Constructor.

        Args:
            n_envs (int): number of parallel environments;
            backend (str): name of the array backend the parsed arrays are built in;
            device (str, None): device the parsed arrays are placed on;
            vectorized (bool, None): whether the step information is provided in vectorized form (a list of dicts or a
                dict of arrays, one entry per environment) rather than as a single dict. If None, it defaults to
                ``n_envs > 1``.

        """
        self._n_envs = n_envs
        self._vectorized = n_envs > 1 if vectorized is None else vectorized
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._device = device

        self._layout = None
        self._records = []
        self._columns = {}
        self._column_rows = {}
        self._pending_steps = 0
        self._key_mapping = {}
        self._flat_keys = {}
        self._shape_mapping = {}
        self._parsed = {}
        self._n_parsed = 0
        self._target_backend = None
        self._target_device = None
        self._source = None
        self._mask = None

        self._add_all_save_attr()

    def __add__(self, other):
        """
        Combine two StepInfo objects into a new one.

        Args:
            other (StepInfo): StepInfo which will be combined with this one.

        Returns:
            A new StepInfo holding the content of both, in order.

        """
        info = self.copy()
        info += other

        return info

    def __iadd__(self, other):
        """
        In-place append: extend this object with the content of another StepInfo.

        Args:
            other (StepInfo): StepInfo whose content will be appended.

        """
        assert self._n_envs == other.n_envs

        self._resolve_source()
        other._resolve_source()

        if self._can_take_pending(other):
            if other._parsed or other._n_parsed:
                self._consolidate()
                self._parsed = self._concatenate_dictionary(self._parsed, other._parsed, self._array_backend,
                                                            other._array_backend, self._n_parsed, other._n_parsed)
                self._n_parsed += other._n_parsed
            self._take_pending(other)
        else:
            own = self.parse()
            theirs = other.parse()
            self._parsed = self._concatenate_dictionary(own, theirs, self._array_backend, other._array_backend,
                                                        self._n_parsed, other._n_parsed)
            self._n_parsed += other._n_parsed

        self._key_mapping.update(other._key_mapping)
        self._flat_keys.update(other._flat_keys)
        self._shape_mapping.update(other._shape_mapping)

        return self

    def append(self, info):
        """
        Append the step information of one step.

        Args:
            info (dict or list): the information to append, either a list holding one dictionary per environment,
                or a single dictionary whose values carry the environment dimension.

        """
        if self._vectorized:
            assert isinstance(info, (dict, list))
        else:
            assert isinstance(info, dict)

        if self._layout is None:
            self._layout = 'records' if isinstance(info, list) else 'columns'

        if self._layout == 'records':
            self._records.append(info)
        else:
            self._append_columns(info)

        self._pending_steps += 1

    def parse(self, to=None):
        """
        Turn the appended step information into a flat dictionary of arrays, one entry per key, merging it with
        the information parsed by previous calls.

        Args:
            to (str, None): backend of the resulting arrays, ``'torch'`` or ``'numpy'``; defaults to the backend
                currently in use.

        Returns:
            A flat dictionary containing an array for every key of the step information. It describes the
            information as of this call and is replaced by the next call that follows an append.

        """
        self._resolve_source()
        self._consolidate()

        if to is None:
            if self._target_backend is not None:
                self._parsed = self._convert_parsed(self._target_backend, self._target_device)
                self._array_backend = ArrayBackend.get_array_backend(self._target_backend)
                self._device = self._target_device
                self._target_backend = None
                self._target_device = None

            return self._parsed

        if to == self._array_backend.get_backend_name():
            return self._parsed

        return self._convert_parsed(to, None)

    def to_backend(self, backend, device=None):
        """
        Copy the step information, to be parsed in another backend.

        Args:
            backend (str): name of the array backend the arrays are built in;
            device (str, None): device the arrays are placed on, or ``None`` for the default one.

        Returns:
            A new StepInfo holding the same content, parsed in the given backend.

        """
        info = self.copy()

        if backend != self._array_backend.get_backend_name() or device != self._device:
            info._target_backend = backend
            info._target_device = device
        else:
            info._target_backend = None
            info._target_device = None

        return info

    def flatten(self, mask=None):
        """
        Combine the environment dimension of the stored information into the step one, keeping only the entries
        selected by the mask.

        Args:
            mask (Array, None): boolean mask selecting, for every step, the environments to keep; ``None`` keeps
                all of them.

        Returns:
            A single-environment StepInfo holding the selected entries.

        """
        assert self._vectorized

        backend = self._array_backend.get_backend_name() if self._target_backend is None else self._target_backend
        device = self._device if self._target_backend is None else self._target_device

        info = StepInfo(1, backend, device)
        info._source = self.copy()
        info._mask = mask

        return info

    def get_view(self, index, copy=False):
        """
        Select a subset of the stored steps.

        Args:
            index (int, slice, ndarray, tensor): the steps the result should contain;
            copy (bool, False): whether the content should be copied rather than shared.

        Returns:
            A new StepInfo holding only the selected steps.

        """
        if isinstance(index, int):
            index = slice(index, index + 1 if index != -1 else None)

        data = self.parse()

        info = StepInfo(self._n_envs, self._array_backend.get_backend_name(), self._device,
                        vectorized=self._vectorized)
        info._key_mapping = self._key_mapping.copy()
        info._flat_keys = self._flat_keys.copy()
        info._shape_mapping = self._shape_mapping.copy()

        if self._array_backend.get_backend_name() == 'list':
            info._parsed = {key: self._select_list(value, index, copy) for key, value in data.items()}
        elif not copy:
            info._parsed = {key: value[index, ...] for key, value in data.items()}
        else:
            for key, value in data.items():
                value = value[index, ...]
                info._parsed[key] = self._array_backend.empty(value.shape, self._device)
                info._parsed[key][:] = value

        info._n_parsed = self._selection_length(index, info._parsed)

        return info

    def copy(self):
        """
        Returns:
            A new StepInfo holding the same content, sharing the stored arrays.

        """
        info = StepInfo(self._n_envs, self._array_backend.get_backend_name(), self._device,
                        vectorized=self._vectorized)
        info._layout = self._layout
        info._records = self._records.copy()
        info._columns = {key: value.copy() for key, value in self._columns.items()}
        info._column_rows = {key: value.copy() for key, value in self._column_rows.items()}
        info._pending_steps = self._pending_steps
        info._key_mapping = self._key_mapping.copy()
        info._flat_keys = self._flat_keys.copy()
        info._shape_mapping = self._shape_mapping.copy()
        info._parsed = self._parsed.copy()
        info._n_parsed = self._n_parsed
        info._target_backend = self._target_backend
        info._target_device = self._target_device
        info._source = self._source
        info._mask = self._mask

        return info

    def clear(self):
        """
        Drop all the stored information.

        """
        self._layout = None
        self._parsed = {}
        self._n_parsed = 0
        self._key_mapping = {}
        self._flat_keys = {}
        self._shape_mapping = {}
        self._source = None
        self._mask = None
        self._clear_pending()

    def drop_before(self, first_step):
        """
        Drop the steps before the given one.

        Args:
            first_step (int): index of the first step to keep.

        """
        assert self._source is None

        n_parsed = self._parsed_steps()

        if first_step < n_parsed:
            self._parsed = {key: value[first_step:] for key, value in self._parsed.items()}
            self._n_parsed -= first_step
            return

        self._parsed = {}
        self._n_parsed = 0
        dropped = first_step - n_parsed

        if self._layout == 'records':
            del self._records[:dropped]
        elif self._layout == 'columns':
            self._drop_columns_before(dropped)

        self._pending_steps = max(self._pending_steps - dropped, 0)

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def n_steps(self):
        steps = self._n_parsed + self._pending_steps

        if self._source is not None:
            if self._mask is None:
                steps += self._source.n_steps * self._source.n_envs
            else:
                steps += int(ArrayBackend.get_array_backend_from(self._mask).sum(self._mask))

        return steps

    def _add_all_save_attr(self):
        self._add_save_attr(
            _n_envs='primitive',
            _vectorized='primitive',
            _array_backend='primitive',
            _device='primitive',
            _layout='primitive',
            _records='primitive',
            _columns='primitive',
            _column_rows='primitive',
            _pending_steps='primitive',
            _key_mapping='primitive',
            _flat_keys='primitive',
            _shape_mapping='primitive',
            _parsed='primitive',
            _n_parsed='primitive',
            _target_backend='primitive',
            _target_device='primitive',
            _source='mushroom',
            _mask='primitive'
        )

    def _drop_columns_before(self, dropped):
        """
        Drop the values reported before the given step, forgetting the keys left without any.

        Args:
            dropped (int): number of steps dropped from the front.

        """
        for key in list(self._columns):
            kept = [(value, row - dropped) for value, row in zip(self._columns[key], self._column_rows[key])
                    if row >= dropped]

            if kept:
                self._columns[key] = [value for value, _ in kept]
                self._column_rows[key] = [row for _, row in kept]
            else:
                del self._columns[key]
                del self._column_rows[key]
                del self._flat_keys[tuple(self._key_mapping[key])]
                del self._key_mapping[key]
                del self._shape_mapping[key]

    def _can_take_pending(self, other):
        """
        Check whether the unparsed rows of another step information can be appended as they are.

        Args:
            other (StepInfo): StepInfo whose rows are to be appended.

        Returns:
            Whether the rows the other object has not parsed yet can be taken over as they are.

        """
        return (self._array_backend is other._array_backend
                and self._device == other._device
                and self._target_backend == other._target_backend
                and self._target_device == other._target_device
                and (self._layout is None or other._layout is None or self._layout == other._layout))

    def _take_pending(self, other):
        """
        Append the rows the other object has not parsed yet, leaving them unparsed.

        Args:
            other (StepInfo): StepInfo whose rows will be appended.

        """
        if other._layout is None:
            return

        if self._layout is None:
            self._layout = other._layout

        if self._layout == 'records':
            self._records += other._records
        else:
            for key, values in other._columns.items():
                self._columns.setdefault(key, []).extend(values)
                self._column_rows.setdefault(key, []).extend([row + self._pending_steps
                                                              for row in other._column_rows[key]])

        self._pending_steps += other._pending_steps

    def _clear_pending(self):
        self._records = []
        self._columns = {}
        self._column_rows = {}
        self._pending_steps = 0

    def _resolve_source(self):
        """
        Apply the selection a :meth:`flatten` call recorded, if there is one.

        """
        if self._source is not None:
            source = self._source
            mask = self._mask
            self._source = None
            self._mask = None

            data = source.parse()
            self._key_mapping = {**source._key_mapping, **self._key_mapping}
            self._flat_keys = {**source._flat_keys, **self._flat_keys}
            self._shape_mapping = {**source._shape_mapping, **self._shape_mapping}

            if mask is None:
                self._parsed = {key: self._array_backend.flatten(value) for key, value in data.items()}
                self._n_parsed = source.n_steps * source.n_envs
            else:
                mask = ArrayBackend.convert_mask(mask, to=self._array_backend.get_backend_name(),
                                                 device=self._array_backend.check_device(self._device))
                self._parsed = {key: self._array_backend.pack_padded_sequence(value, mask)
                                for key, value in data.items()}
                self._n_parsed = int(ArrayBackend.get_array_backend_from(mask).sum(mask))

    def _consolidate(self):
        """
        Merge the steps appended since the last call into the arrays holding the ones before them, leaving
        the result in the backend the content is stored in.

        """
        if self._pending_steps:
            if self._layout == 'records':
                self._discover_record_keys()

            n_parsed = self._parsed_steps()
            total = n_parsed + self._pending_steps
            size = (total, self._n_envs) if self._vectorized else (total,)

            output = {key: self._array_backend.full(size + self._shape_mapping[key],
                                                    self._array_backend.none(), self._device)
                      for key in self._key_mapping}

            for key in output:
                if key in self._parsed:
                    output[key][:n_parsed] = self._parsed[key]

            if self._layout == 'records':
                self._fill_from_records(output, n_parsed)
            elif self._layout == 'columns':
                self._fill_from_columns(output, n_parsed)

            self._parsed = output
            self._n_parsed = total
            self._clear_pending()

    def _convert_parsed(self, to, device):
        """
        Convert the parsed arrays to another backend.

        Args:
            to (str): name of the array backend the resulting arrays are built in;
            device (str, None): device the resulting arrays are placed on, or ``None`` for the default one.

        Returns:
            A flat dictionary holding the consolidated arrays converted to the given backend and device.

        """
        device = ArrayBackend.get_array_backend(to).check_device(device)

        return {key: ArrayBackend.convert(value, to=to, backend=self._array_backend, device=device)
                for key, value in self._parsed.items()}

    def _parsed_steps(self):
        return self._n_parsed

    def _discover_record_keys(self):
        for step_data in self._records:
            for env_data in step_data:
                assert isinstance(env_data, dict)
                self._register_keys(env_data, True)

    def _fill_from_records(self, output, offset):
        none_value = self._array_backend.none()
        for step, step_data in enumerate(self._records):
            index = offset + step
            for key, key_path in self._key_mapping.items():
                for env in range(min(len(step_data), self._n_envs)):
                    value = self._find_element_by_key_path(step_data[env], key_path)
                    output[key][index][env] = none_value if value is None else value

    def _fill_from_columns(self, output, offset):
        none_value = self._array_backend.none()
        for key, values in self._columns.items():
            rows = self._column_rows[key]
            for value, row in zip(values, rows):
                if value is None:
                    value = none_value
                if value is none_value or not self._vectorized:
                    output[key][offset + row] = value
                else:
                    output[key][offset + row] = value[:self._n_envs]

    def _append_columns(self, info):
        stack = [(info, [])]

        while stack:
            structure_element, parent_keys = stack.pop()
            assert isinstance(structure_element, dict)

            for key, value in structure_element.items():
                key_path = parent_keys + [key]

                if isinstance(value, dict):
                    stack.append((value, key_path))
                else:
                    flat_key = self._register_key(key_path, value, not self._vectorized)
                    self._columns.setdefault(flat_key, []).append(value)
                    self._column_rows.setdefault(flat_key, []).append(self._pending_steps)

    def _register_keys(self, template, single_env):
        """
        Register every key of the given dictionary, descending into the nested ones.

        Args:
            template (dict): dictionary to extract the keys from;
            single_env (bool): whether the template holds the data of a single environment.

        """
        assert isinstance(template, dict)

        stack = [(template, [])]

        while stack:
            structure_element, parent_keys = stack.pop()
            assert isinstance(structure_element, dict)

            for key, value in structure_element.items():
                key_path = parent_keys + [key]

                if isinstance(value, dict):
                    stack.append((value, key_path))
                else:
                    self._register_key(key_path, value, single_env)

    def _register_key(self, key_path, value, single_env):
        """
        Record the flat key a key path maps to, together with the shape of its values.

        Args:
            key_path (list): the keys leading to the value;
            value: the value reported for that key path;
            single_env (bool): whether the value belongs to a single environment.

        Returns:
            The flat key the key path maps to.

        """
        path = tuple(key_path)
        key = self._flat_keys.get(path)

        if key is None:
            key = "_".join(str(element) for element in key_path)
            self._flat_keys[path] = key
            self._key_mapping[key] = key_path
            self._store_array_shape(key, value, single_env)

        return key

    def _store_array_shape(self, key, value, single_env):
        """
        Stores the shape of the value. If value does not have a shape, an empty tuple is stored.

        Args:
            key (str): Dictionary key.
            value (Array, Number): Variable whose shape should be saved
            single_env (bool): whether the value belongs to a single environment.
        """
        if isinstance(value, numbers.Number) or value is None:
            self._shape_mapping[key] = ()
        else:
            shape = self._array_backend.shape(value)
            self._shape_mapping[key] = shape[1:] if not single_env else shape

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

    def _concatenate_array(self, array1, array2, intended_length_array1, intended_length_array2, array1_backend,
                           array2_backend):
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
        device = array1_backend.check_device(self._device)

        if array1 is None:
            shape = (intended_length_array1,) + array2_backend.shape(array2)[1:]
            array1 = array1_backend.full(shape, array1_backend.none(), device)
        if array2 is None:
            shape = (intended_length_array2, ) + array1_backend.shape(array1)[1:]
            array2 = array2_backend.full(shape, array2_backend.none())
        array2 = array1_backend.convert(array2, backend=array2_backend, device=device)
        return array1_backend.concatenate((array1, array2))

    def _concatenate_dictionary(self, dict1, dict2, backend1, backend2, length1, length2):
        """
        Concatenate dict1 with dict2.

        Args:
            dict1 (dict): Flat dictionary containing arrays of backend1
            dict2 (dict): Flat dictionary containing arrays of backend2
            backend1 (ArrayBackend): Backend of arrays in dict1.
            backend2 (ArrayBackend): Backend of arrays in dict2.
            length1 (int): Number of steps dict1 accounts for.
            length2 (int): Number of steps dict2 accounts for.

        Returns
            dict: Concatenation of dict1 and dict2
        """
        if not dict1 and not length1:
            device = backend1.check_device(self._device)
            return {key: backend1.convert(value, backend=backend2, device=device)
                    for key, value in dict2.items()}
        if not dict2 and not length2:
            return dict1

        r = {}

        for key in dict1.keys() | dict2.keys():
            array1 = dict1[key] if key in dict1 else None
            array2 = dict2[key] if key in dict2 else None
            r[key] = self._concatenate_array(array1, array2, length1, length2, backend1, backend2)
        return r

    def _selection_length(self, index, selected):
        """
        Count the steps held by a selection.

        Args:
            index (int, slice, ndarray, tensor): the selection applied to the stored steps;
            selected (dict): the selected content.

        Returns:
            The number of steps the selection holds.

        """
        if selected:
            return self._array_backend.shape(next(iter(selected.values())))[0]
        if isinstance(index, slice):
            return len(range(*index.indices(self._n_parsed)))
        if isinstance(index, int):
            return 1

        backend = ArrayBackend.get_array_backend_from(index)
        dtype = getattr(index, 'dtype', None)

        if dtype is not None and dtype == backend.to_backend_dtype(bool):
            return int(backend.sum(index))

        return len(index)

    @staticmethod
    def _select_list(value, index, copy):
        if isinstance(index, (int, slice)):
            selected = value[index]
        else:
            selected = [value[i] for i in index]
        return deepcopy(selected) if copy else selected
