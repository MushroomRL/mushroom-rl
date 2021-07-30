Habitat and iGibson support many realistic scenes as environment for the agent.
By default the agent's observations are RGB images, but RGBD, sensory data, and
other information can also be used.

> If you have previous versions of iGibson or Habitat already installed, we
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

The scene details are defined in a YAML file, that needs to be passed to the agent.
See `<IGIBSON PATH>/igibson/test/test_house.YAML` for an example.


## Habitat Installation
Follow the [official guide](https://github.com/facebookresearch/habitat-lab/#installation)
and do a **full install** with `habitat_baselines`.
Then you can download interactive datasets following
[this](https://github.com/facebookresearch/habitat-lab#data) and
[this](https://github.com/facebookresearch/habitat-lab#task-datasets).
If you need to download other datasets, you can use
[this utility](https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/utils/datasets_download.py).

### Basic Usage of Habitat
When you create a `Habitat` environment, you need to pass a wrapper name and two
YAML files: `Habitat(wrapper, config_file, base_config_file)`.
* The wrapper has to be among the ones defined in
`<MUSHROOM_RL PATH>/mushroom-rl/environments/habitat_env.py`, and takes care of
converting actions and observations in a gym-like format. If your task / robot
requires it, you may need to define new wrappers.
* The YAML files define every detail: the Habitat environment, the scene, the
sensors available to the robot, the rewards, the action discretization, and any
additional information you may need. The second YAML file is optional, and
overwrites whatever was already defined in the first YAML.
> If you use YAMLs from `habitat-lab`, check if they define a YAML for
BASE_TASK_CONFIG_PATH. If they do, you need to pass it as `base_config_file` to
`Habitat()`. `habitat-lab` YAMLs, in fact, use relative paths, and calling them
from outside its root folder will cause errors.

* If you use a dataset, be sure that the path defined in the YAML file is correct,
especially if you use relative paths. `habitat-lab` YAMLs use relative paths, so
be careful with that. By default, the path defined in the YAML file will be
relative to where you launched the python code. See the navigation example below
for more details.

### Rearrange Task Example
* Download assets and the ReplicaCAD datasets
```
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data
```
* For this task we use `<HABITAT_LAB PATH>/habitat_baselines/config/rearrange/ddppo_rearrangepick.yaml`.
This YAML defines `BASE_TASK_CONFIG_PATH: configs/tasks/rearrangepick_replica_cad.yaml`,
and since this is a relative path we need to overwrite it by passing its absolute path
as `base_config_file` argument to `Habitat()`.
* Then, `rearrangepick_replica_cad.yaml` defines the dataset to be used, and
this is in `<HABITAT_LAB PATH>`. However, since the path defined is relative
to where we launch our code, we need to make a link to the data folder. If you
launch `habitat_rearrange_sac.py` from its example folder, run
```
ln -s <HABITAT_LAB PATH>/data/ <MUSHROOM_RL PATH>/mushroom-rl/examples/habitat
```
* Finally, you can launch `python habitat_rearrange_sac.py`.

### Navigation Task Example
* Download and extract Replica scenes
> WARNING! The dataset is very large!

```
sudo apt-get install pigz
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh replica-path
```
* For this task we only use the custom YAML file `pointnav_apartment-0.yaml`.
* `DATA_PATH: "replica_{split}_apartment-0.json.gz"` defines the JSON file with
some scene details, such as the agent's initial position and orientation.
The `{split}` value is defined in the `SPLIT` key.
> If you want to try new positions, you can sample some from the set of the
scene's navigable points. After initializing a `habitat` environment, for example
`mdp = Habitat(...)`, run `mdp.env._env._sim.sample_navigable_point()`.

* `SCENES_DIR: "Replica-Dataset/replica-path/apartment_0"` defines the scene.
As said before, this path is relative to where you launch the script, thus we
need to link the Replica folder. If you launch `habitat_nav_dqn.py` from its example folder, run
```
ln -s <PATH TO>/Replica-Dataset/ <MUSHROOM_RL PATH>/mushroom-rl/examples/habitat
```
* Finally, you can launch `python habitat_nav_dqn.py`.
