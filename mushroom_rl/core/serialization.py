import sys
import json
import torch
import pickle
import numpy as np

from copy import deepcopy
from pathlib import Path, PurePath

if sys.version_info >= (3, 7):
    from zipfile import ZipFile
else:
    from zipfile37 import ZipFile


class Serializable(object):
    def save(self, path, full_save=True):
        """
        Serialize and save the agent to the given path on disk.

        Args:
            path (Path, string): Relative or absolute path to the agents save
                location.
            full_save (bool): Flag to specify the amount of data to save for mushroom data structures

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ZipFile(path, 'w') as zip_file:
            agent_config = dict(
                type=type(self),
                save_attributes=self._save_attributes
            )

            self._save_pickle(zip_file, 'config', agent_config)

            for att, method in self._save_attributes.items():
                attribute = getattr(self, att) if hasattr(self, att) else None

                if attribute is not None:
                    if hasattr(self, '_save_{}'.format(method)):
                        save_method = getattr(self, '_save_{}'.format(method))
                        file_name = "{}.{}".format(att, method)
                        save_method(zip_file, file_name, attribute, full_save=full_save)
                    else:
                        raise NotImplementedError(
                            "Method _save_{} is not implemented for class '{}'".
                                format(method, self.__class__.__name__)
                        )

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
            print(zip_file.namelist())
            type, save_attributes = cls._load_pickle(zip_file, 'config').values()

            loaded_object = type.__new__(type)

            for att, method in save_attributes.items():
                file_name = '{}.{}'.format(att, method)

                if file_name in zip_file.namelist():
                    load_method = getattr(cls, '_load_{}'.format(method))
                    if load_method is None:
                        raise NotImplementedError('Method _load_{} is not'
                                                  'implemented'.format(method))
                    att_val = load_method(zip_file, file_name)
                    setattr(loaded_object, att, att_val)
                else:
                    print('att', att, 'named', file_name, 'not in zip')
                    setattr(loaded_object, att, None)

        loaded_object._post_load()

        return loaded_object

    def copy(self):
        """
        Returns:
             A deepcopy of the agent.

        """
        return deepcopy(self)

    def _add_save_attr(self, **attr_dict):
        """
        Add attributes that should be saved for an agent.

        Args:
            attr_dict (dict): dictionary of attributes mapped to the method that
                should be used to save and load them.

        """
        if not hasattr(self, '_save_attributes'):
            self._save_attributes = dict(_save_attributes='json')
        self._save_attributes.update(attr_dict)

    def _post_load(self):
        """
        This method can be overwritten to implement logic that is executed after
        the loading of the agent.

        """
        pass

    @staticmethod
    def _load_pickle(zip_file, name):
        with zip_file.open(name, 'r') as f:
            return pickle.load(f)

    @staticmethod
    def _load_numpy(zip_file, name):
        print('loading ', name)
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
        with zip_file.open(name, 'f') as f:
            Serializable.load(f)

    @staticmethod
    def _save_pickle(zip_file, name, obj, **_):
        with zip_file.open(name, 'w') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _save_numpy(zip_file, name, obj, **_):
        with zip_file.open(name, 'w') as f:
            np.save(f, obj)

    @staticmethod
    def _save_torch(zip_file, name, obj, **_):
        with zip_file.open(name, 'w') as f:
            torch.save(obj, f)

    @staticmethod
    def _save_json(zip_file, name, obj, **_):
        with zip_file.open(name, 'w') as f:
            string = json.dumps(obj)
            f.write(string.encode('utf8'))

    @staticmethod
    def _save_mushroom(zip_file, name, obj, full_save, **_):
        with zip_file.open(name, 'w') as f:
            obj.save(f, full_save=full_save)