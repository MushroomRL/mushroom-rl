import sys
import json
import torch
import pickle
import numpy as np

from copy import deepcopy
from pathlib import Path

if sys.version_info >= (3, 7):
    from zipfile import ZipFile
else:
    from zipfile37 import ZipFile


class Serializable(object):
    """
    Interface to implement serialization of a MushroomRL object.
    This provide load and save functionality to save the object in a zip file.
    It is possible to save the state of the agent with different levels of

    """
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
        config_path = Serializable._append_folder(folder, 'config')

        try:
            object_type, save_attributes, primitive_dictionary = \
                cls._load_pickle(zip_file, config_path).values()
        except KeyError:
            return None

        if object_type is list:
            return cls._load_list(zip_file, folder, primitive_dictionary['len'])
        else:
            loaded_object = object_type.__new__(object_type)
            setattr(loaded_object, '_save_attributes', save_attributes)

            for att, method in save_attributes.items():
                mandatory = not method.endswith('!')
                method = method[:-1] if not mandatory else method
                file_name = Serializable._append_folder(
                    folder, '{}.{}'.format(att, method)
                )

                if method == 'primitive' and att in primitive_dictionary:
                    setattr(loaded_object, att, primitive_dictionary[att])
                elif file_name in zip_file.namelist() or \
                        (method == 'mushroom' and mandatory):
                    load_method = getattr(cls, '_load_{}'.format(method))
                    if load_method is None:
                        raise NotImplementedError('Method _load_{} is not'
                                                  'implemented'.format(method))
                    att_val = load_method(zip_file, file_name)
                    setattr(loaded_object, att, att_val)

                else:
                    setattr(loaded_object, att, None)

            loaded_object._post_load()

            return loaded_object

    @classmethod
    def _load_list(self, zip_file, folder, length):
        loaded_list = list()

        for i in range(length):
            element_folder = Serializable._append_folder(folder, str(i))
            loaded_element = Serializable.load_zip(zip_file, element_folder)
            loaded_list.append(loaded_element)

        return loaded_list

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
        classes that implement the Serializable interface. All the other methods
        use the library named.
        If a "!" character is added at the end of the method, the field will be
        saved only if full_save is set to True.

        Args:
            **attr_dict: dictionary of attributes mapped to the method
                that should be used to save and load them.

        """
        if not hasattr(self, '_save_attributes'):
            self._save_attributes = dict()
        self._save_attributes.update(attr_dict)

    def _post_load(self):
        """
        This method can be overwritten to implement logic that is executed
        after the loading of the agent.

        """
        pass

    @staticmethod
    def _append_folder(folder, name):
        if folder:
           return folder + '/' + name
        else:
           return name

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
            return torch.load(f)

    @staticmethod
    def _load_json(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_mushroom(zip_file, name):
        return Serializable.load_zip(zip_file, name)

    @staticmethod
    def _save_pickle(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    @staticmethod
    def _save_numpy(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            np.save(f, obj)

    @staticmethod
    def _save_torch(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            torch.save(obj, f)

    @staticmethod
    def _save_json(zip_file, name, obj, folder, **_):
        path = Serializable._append_folder(folder, name)
        with zip_file.open(path, 'w') as f:
            string = json.dumps(obj)
            f.write(string.encode('utf8'))

    @staticmethod
    def _save_mushroom(zip_file, name, obj, folder, full_save):
        new_folder = Serializable._append_folder(folder, name)
        if isinstance(obj, list):
            config_data = dict(
                type=list,
                save_attributes=dict(),
                primitive_dictionary=dict(len=len(obj))
            )

            Serializable._save_pickle(zip_file, 'config', config_data, folder=new_folder)
            for i, element in enumerate(obj):
                element_folder = Serializable._append_folder(new_folder, str(i))
                element.save_zip(zip_file, full_save=full_save, folder=element_folder)
        else:
            obj.save_zip(zip_file, full_save=full_save, folder=new_folder)

    @staticmethod
    def _get_serialization_method(class_name):
        if issubclass(class_name, Serializable):
            return 'mushroom'
        else:
            return 'pickle'

