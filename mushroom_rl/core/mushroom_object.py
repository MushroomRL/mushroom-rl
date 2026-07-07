import json
import torch
import pickle
import numpy as np

from copy import deepcopy
from pathlib import Path

from mushroom_rl.utils.torch_utils import TorchUtils

from zipfile import ZipFile
import inspect


class MushroomObject(object):
    """
    Interface to implement serialization and logging of a MushroomRL object.

    It provides save and load functionality to store the object in a zip file: subclasses declare which
    attributes to persist with ``_add_save_attr``, and ``full_save`` selects how much of the state is
    saved.

    It also provides logging functionality: a logger is attached with ``set_logger`` and forwarded to the
    loggable children declared with ``_add_logger_attr``, so that the relevant quantities of an object and
    of its sub-objects are logged under a hierarchy of metric names.

    """

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)

        obj._save_attributes = dict()
        obj._logger_attributes = dict()
        obj._logger = None
        obj._log_prefix = None
        obj._log_label = None

        return obj

    def save(self, path, full_save=False):
        """
        Serialize and save the object to the given path on disk.

        Args:
            path (Path, str): Relative or absolute path to the object save
                location;
            full_save (bool): Flag to specify the amount of data to save for
                MushroomRL data structures.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ZipFile(path, 'w') as zip_file:
            self.save_zip(zip_file, full_save)

    def save_zip(self, zip_file, full_save, folder=''):
        """
        Serialize and save the agent to the given path on disk.

        Args:
            zip_file (ZipFile): ZipFile where te object needs to be saved;
            full_save (bool): flag to specify the amount of data to save for
                MushroomRL data structures;
            folder (string, ''): subfolder to be used by the save method.
        """
        primitive_dictionary = dict()

        for att, method in self._save_attributes.items():

            if not method.endswith('!') or full_save:
                method = method[:-1] if method.endswith('!') else method
                attribute = getattr(self, att) if hasattr(self, att) else None

                if attribute is not None:
                    if method == 'primitive':
                        primitive_dictionary[att] = attribute
                    elif method == 'none':
                        pass
                    elif hasattr(self, '_save_{}'.format(method)):
                        save_method = getattr(self, '_save_{}'.format(method))
                        file_name = "{}.{}".format(att, method)
                        save_method(zip_file, file_name, attribute,
                                    full_save=full_save, folder=folder)
                    else:
                        raise NotImplementedError(
                            "Method _save_{} is not implemented for class '{}'".
                                format(method, self.__class__.__name__)
                        )

        config_data = dict(
            type=type(self),
            save_attributes=self._save_attributes,
            logger_attributes=self._logger_attributes,
            primitive_dictionary=primitive_dictionary
        )

        self._save_pickle(zip_file, 'config', config_data, folder=folder)

    @classmethod
    def load(cls, path):
        """
        Load and deserialize the agent from the given location on disk.

        Args:
            path (Path, string): Relative or absolute path to the agents save
                location.

        Returns:
            The loaded agent.

        """
        path = Path(path)
        if not path.exists():
            raise ValueError("Path to load agent is not valid")

        with ZipFile(path, 'r') as zip_file:
            loaded_object = cls.load_zip(zip_file)

        return loaded_object

    @classmethod
    def load_zip(cls, zip_file, folder=''):
        config_path = MushroomObject._append_folder(folder, 'config')

        try:
            config = cls._load_pickle(zip_file, config_path)
            object_type = config['type']
            save_attributes = config['save_attributes']
            logger_attributes = config['logger_attributes']
            primitive_dictionary = config['primitive_dictionary']
        except KeyError:
            return None

        if object_type is list:
            return cls._load_list(zip_file, folder, primitive_dictionary['len'],
                                  primitive_dictionary['method'])
        else:
            loaded_object = object_type.__new__(object_type)
            setattr(loaded_object, '_save_attributes', save_attributes)
            setattr(loaded_object, '_logger_attributes', logger_attributes)

            for att, method in save_attributes.items():
                mandatory = not method.endswith('!')
                method = method[:-1] if not mandatory else method
                file_name = MushroomObject._append_folder(
                    folder, '{}.{}'.format(att, method)
                )

                if method == 'primitive' and att in primitive_dictionary:
                    setattr(loaded_object, att, primitive_dictionary[att])
                elif file_name in zip_file.namelist():
                    load_method = getattr(cls, '_load_{}'.format(method))
                    att_val = load_method(zip_file, file_name)
                    setattr(loaded_object, att, att_val)
                elif MushroomObject._is_saved_list(zip_file, file_name):
                    att_val = MushroomObject.load_zip(zip_file, file_name)
                    setattr(loaded_object, att, att_val)
                else:
                    setattr(loaded_object, att, None)

            loaded_object._post_load()

            return loaded_object

    def set_logger(self, logger, prefix=None, label=None):
        """
        Attach a logger to the object so that its relevant quantities are logged. The ``prefix``
        groups the logged metrics (e.g. ``critic`` produces ``critic/loss``); the ``label`` overrides
        the default metric name of a single-value object. The logger is then forwarded to every
        loggable child registered with ``_add_logger_attr``, with the child group joined to ``prefix``.

        Args:
            logger (Logger): the logger to be used by the object;
            prefix (str, None): optional group prepended to the logged metric names;
            label (str, None): optional metric name override for single-value objects.

        """
        self._logger = logger
        self._log_prefix = prefix
        self._log_label = label

        for attr, (group, child_label) in self._logger_attributes.items():
            child = getattr(self, attr, None)
            if child is not None:
                child.set_logger(logger, self._join_prefix(prefix, group), child_label)

    def copy(self):
        """
        Returns:
             A deepcopy of the agent.

        """
        return deepcopy(self)

    def _add_save_attr(self, **attr_dict):
        """
        Add attributes that should be saved for an agent.
        For every attribute, it is necessary to specify the method to be used to
        save and load.
        Available methods are: numpy, mushroom, torch, json, pickle, primitive
        and none. The primitive method can be used to store primitive attributes,
        while the none method always skip the attribute, but ensure that it is
        initialized to None after the load. The mushroom method can be used with
        classes that implement the MushroomObject interface. All the other methods
        use the library named.
        If a "!" character is added at the end of the method, the field will be
        saved only if full_save is set to True.

        Args:
            **attr_dict: dictionary of attributes mapped to the method
                that should be used to save and load them.

        """
        self._save_attributes.update(attr_dict)

    def _add_logger_attr(self, *attrs, group=None, **labels):
        """
        Register loggable child attributes so that ``set_logger`` forwards the logger to each of them,
        all grouped under the optional ``group`` prefix. Attributes passed positionally use their default
        metric name (``loss`` for an approximator, ``value`` for a parameter), while attributes passed as
        keywords map to an explicit metric name. For example ``_add_logger_attr('_V', group='critic')`` logs
        the approximator under ``critic/loss``, and ``_add_logger_attr(_epsilon='epsilon', group='policy')``
        logs the parameter under ``policy/epsilon``. The registry is saved, so the forwarding keeps working
        on a loaded object.

        Args:
            *attrs: attribute names that use their child default metric name;
            group (str, None): optional group prefix shared by all the registered children;
            **labels: mapping from attribute name to its explicit metric label.

        """
        for attr in attrs:
            self._logger_attributes[attr] = (group, None)
        for attr, label in labels.items():
            self._logger_attributes[attr] = (group, label)

    def _post_load(self):
        """
        This method can be overwritten to implement logic that is executed
        after the loading of the agent.

        """
        pass

    @staticmethod
    def _join_prefix(prefix, group):
        """
        Join a parent ``prefix`` and a child ``group`` into a logging prefix using ``'/'`` as separator,
        ignoring the empty ones. Returns ``None`` if both are empty, so that the prefix composes cleanly
        through nested objects.

        """
        if prefix and group:
            return prefix + '/' + group
        return prefix or group

    @staticmethod
    def _append_folder(folder, name):
        if folder:
           return folder + '/' + name
        else:
           return name

    @staticmethod
    def _is_saved_list(zip_file, name):
        return MushroomObject._append_folder(name, 'config') in zip_file.namelist()

    @staticmethod
    def _load_list(zip_file, folder, length, method):
        loaded_list = list()

        load_method = getattr(MushroomObject, '_load_{}'.format(method))

        for i in range(length):
            element_name = MushroomObject._append_folder(folder, str(i))
            if MushroomObject._is_saved_list(zip_file, element_name):
                loaded_list.append(MushroomObject.load_zip(zip_file, element_name))
            else:
                loaded_list.append(load_method(zip_file, element_name))

        return loaded_list

    @staticmethod
    def _load_pickle(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return pickle.load(f)

    @staticmethod
    def _load_numpy(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return np.load(f)

    @staticmethod
    def _load_torch(zip_file, name):
        with zip_file.open(name, 'r') as f:
            #modern versions of torch require weights_only to be explicitly set to False
            if 'weights_only' in inspect.signature(torch.load).parameters:
                return torch.load(f, map_location=TorchUtils.get_device(), weights_only=False)
            else:
                return torch.load(f, map_location=TorchUtils.get_device())
    @staticmethod
    def _load_json(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_mushroom(zip_file, name):
        return MushroomObject.load_zip(zip_file, name)

    @staticmethod
    def _save_pickle(zip_file, name, obj, folder, **_):
        path = MushroomObject._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    @staticmethod
    def _save_numpy(zip_file, name, obj, folder, **_):
        if isinstance(obj, list):
            MushroomObject._save_list(zip_file, name, obj, folder, 'numpy')
        else:
            path = MushroomObject._append_folder(folder, name)
            with zip_file.open(path, 'w') as f:
                np.save(f, obj)

    @staticmethod
    def _save_torch(zip_file, name, obj, folder, **_):
        path = MushroomObject._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            torch.save(obj, f)

    @staticmethod
    def _save_json(zip_file, name, obj, folder, **_):
        path = MushroomObject._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            string = json.dumps(obj)
            f.write(string.encode('utf8'))

    @staticmethod
    def _save_mushroom(zip_file, name, obj, folder, full_save):
        if isinstance(obj, list):
            MushroomObject._save_list(zip_file, name, obj, folder, 'mushroom', full_save)
        else:
            new_folder = MushroomObject._append_folder(folder, name)
            obj.save_zip(zip_file, full_save=full_save, folder=new_folder)

    @staticmethod
    def _save_list(zip_file, name, obj, folder, method, full_save=False):
        new_folder = MushroomObject._append_folder(folder, name)
        config_data = dict(
            type=list,
            save_attributes=dict(),
            logger_attributes=dict(),
            primitive_dictionary=dict(len=len(obj), method=method)
        )
        MushroomObject._save_pickle(zip_file, 'config', config_data, folder=new_folder)

        save_method = getattr(MushroomObject, '_save_{}'.format(method))
        for i, element in enumerate(obj):  # a nested list recurses via the method's own list guard
            save_method(zip_file, str(i), element, folder=new_folder, full_save=full_save)

    @staticmethod
    def _get_serialization_method(class_name):
        if issubclass(class_name, MushroomObject):
            return 'mushroom'
        else:
            return 'pickle'

