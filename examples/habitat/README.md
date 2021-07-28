>> If you already have old versions of Habitat or iGibson installed,
we recommend to do a clean install.


## Habitat Installation
Follow the [official guide](https://github.com/facebookresearch/habitat-lab#installation).
You have to install *the full version* (that incudes `habitat_baselines`) and
[habitat-sim](https://github.com/facebookresearch/habitat-sim/).
Then follow the instructions to run the example, and you will download all
interactive datasets.


## iGibson Installation
Follow the official guide [here](https://github.com/StanfordVL/iGibson).


## Scene Datasets
Habitat and iGibson support many realistic scenes as environment for the agent.

iGibson has its own dataset that can be downloaded and used right away.
Alternatively, you can use third party datasets. The scene is defined a yaml
file, that needs to be passed to the agent. See `igibson/test/test_house.yaml`
for an example. For more details, please see the
[official documentation](http://svl.stanford.edu/igibson/).

For Habitat, you need to download scenes separately. For more details, please
see [here](https://github.com/facebookresearch/habitat-lab#data) and
[here](https://github.com/facebookresearch/habitat-lab#task-datasets).
In `examples/habitat` we use Replica and ReplicaCAD for a navigation and
an interaction demo, respectively. If you have followed `habitat-lab` instructions
and ran its example, you should have all the interactive datasets already installed.
Below, we explain how to use Replica scenes.

### How to Use Replica Scenes
* [Download Replica scenes](https://github.com/facebookresearch/Replica-Dataset).
When you run `./download.sh /path/to/replica_v1`, this path will have to be
set in the yaml file under `DATASET.SCENES_DIR`.
* Scene details, such as the agent's initial position and orientation, are defined
in a json file. This file is usually named `replica-{split}.json`, where `split`
is defined in the yaml file under `DATASET.SPLIT`. You need to pass the json file
to the yaml file under `DATASET.DATA_PATH`.

See `example/habitat_dqn` for an example.
