Habitat and iGibson support many realistic scenes as environment for the agent.
By default the agent's observations are RGB images, but RGBD, sensory data, and
other information can also be used.

>> If you have previous version of iGibson or Habitat already installed, we
recommend to remove them and do clean installs.

## iGibson Installation
Follow the [official guide](http://svl.stanford.edu/igibson/#install_env) and
install its [assets](http://svl.stanford.edu/igibson/docs/assets.html) and
[datasets](http://svl.stanford.edu/igibson/docs/dataset.html).

For our example you need to run
```
python -m igibson.utils.assets_utils --download_assets
python -m igibson.utils.assets_utils --download_demo_data
```

You can also use [third party datasets](https://github.com/StanfordVL/iGibson/tree/master/igibson/utils/data_utils/ext_scene).
The scene details are defined in a yaml file, that needs to be passed to the agent.
See `<IGIBSON PATH>/igibson/test/test_house.yaml` for an example.


## Habitat Installation
Follow the [official guide](https://github.com/facebookresearch/habitat-lab/#installation)
and do a **full install** with `habitat_baselines`.
Then you can install interactive datasets following
[this](https://github.com/facebookresearch/habitat-lab#data) and
[this](https://github.com/facebookresearch/habitat-lab#task-datasets).

For our examples, we need Replica and ReplicaCAD datasets. For ReplicaCAD run
```
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data
```
[this](https://github.com/facebookresearch/habitat-lab#replicacad)

If you need to download other datasets, you can use
[this utility](https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/utils/datasets_download.py).


If you want to suppress Habitat messages run
```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
```


### How to Use Replica Scenes
Download Replica scenes
```
sudo apt-get install pigz
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh replica-path
```
The Replica path will have to be passed to yaml files under `DATASET.SCENES_DIR`
whenever you want to use one of its scenes. In the navigation examples
`pointnav_apartment-0.yaml`, we assume that you have download Replica scenes in
`<MUSHROOM_RL PATH>/examples/habitat/Replica-Dataset/replica-path`.

Scene details, such as the agent's initial position and orientation, are defined
in a json file. This file is usually named `replica-{split}.json`, where `split`
is defined in the yaml file under `DATASET.SPLIT`. You need to pass the json file
to the yaml file under `DATASET.DATA_PATH`.

Scene details, such as the agent's initial position and orientation, are
defined in the replica json file. If you want to try new positions, you can
sample some from the set of the scene's navigable points, accessible by
NavRLEnv._env._sim.sample_navigable_point().
