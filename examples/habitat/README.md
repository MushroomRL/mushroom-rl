## Habitat Installation
* [Install habitat-sim](https://github.com/facebookresearch/habitat-sim/).
It is recommended to install [the nightly build with conda](https://anaconda.org/aihabitat-nightly/habitat-sim).
Otherwise, [install it from souce](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md).
* [Install habitat-lab](https://github.com/facebookresearch/habitat-lab).
The full installation including `habitat_baselines` is required.


## iGibson Installation
Follow the official guide [here](https://github.com/StanfordVL/iGibson).


## Scene Datasets
Habitat and iGibson support many realistic scenes as environment for the agent.

iGibson has its own dataset that can be downloaded and used right away.
Alternatively, you can use third party datasets. The scene is defined a yaml
file, that needs to be passed to the agent. See `<IGIBSON PATH>/igibson/test/test_house.yaml`
for an example. For more details, please see the
[official documentation](http://svl.stanford.edu/igibson/).

For Habitat, you need to download scenes separately. For more details, please
see [here](https://github.com/facebookresearch/habitat-lab#data) and
[here](https://github.com/facebookresearch/habitat-lab#task-datasets).
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
