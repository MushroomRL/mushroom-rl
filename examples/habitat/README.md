Habitat and iGibson support many realistic scenes as environment for the agent.
By default the agent's observations are RGB images, but RGBD, sensory data, and
other information can also be used.

>> If you have previous version of iGibson or Habitat already installed, we
recommend to remove them and do clean installs.

## iGibson Installation
Follow the [official guide](http://svl.stanford.edu/igibson/#install_env) and
install its [assets](http://svl.stanford.edu/igibson/docs/assets.html) and
[datasets](http://svl.stanford.edu/igibson/docs/dataset.html).
For the example you need to run
```
python -m igibson.utils.assets_utils --download_assets
python -m igibson.utils.assets_utils --download_demo_data
```
You can also use [third party datasets](https://github.com/StanfordVL/iGibson/tree/master/igibson/utils/data_utils/ext_scene).
The scene details are defined in a yaml file, that needs to be passed to the agent.
See `<IGIBSON PATH>/igibson/test/test_house.yaml` for an example.


logging.getLogger().setLevel(logging.INFO)


## Habitat Installation
Follow the [official guide](https://github.com/facebookresearch/habitat-lab/#installation)
and do a **full install** with `habitat_baselines`.

For Habitat, you need to download scenes separately. For more details, please
see [here](https://github.com/facebookresearch/habitat-lab#data) and
[here](https://github.com/facebookresearch/habitat-lab#task-datasets).
If you have followed the instructions and ran `python examples/example.py`,
you should have already downloaded all interactive datasets.
If you need to download other datasets, you can use
[this utility](https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/utils/datasets_download.py).

For our example, we need to run
```
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data
```

Unfortunately, Habitat looks for dataset...

```
ln -s /private/home/sparisi/habitat-baselines/habitat-lab/data/ /private/home/sparisi/mushroom-rl/examples/habitat
```

If you want to suppress Habitat messages run:
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet


In `<MUSHROOM_RL PATH>/examples/habitat` we use Replica and ReplicaCAD for
navigation and interaction tasks, respectively. For ReplicaCAD, follow
[this](https://github.com/facebookresearch/habitat-lab#replicacad).
Below, we explain how to use Replica scenes.

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
