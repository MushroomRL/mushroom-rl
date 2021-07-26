## Habitat Installation

1. [Install habitat-sim](https://github.com/facebookresearch/habitat-sim/).
It is recommended to install [the nightly build with conda](https://anaconda.org/aihabitat-nightly/habitat-sim).
Otherwise, [install it from souce](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md).

2. [Install habitat-lab](https://github.com/facebookresearch/habitat-lab).
The full installation including `habitat_baselines` is required.


## iGibson Installation
Follow the official guide [here](https://github.com/StanfordVL/iGibson).


## Scene Datasets
Habitat and iGibson support many realistic scenes as environment for the agent.
iGibson has its own dataset, that can be downloaded and used right away.
Alternatively, you can use third party datasets. Please see the
[official documentation](http://svl.stanford.edu/igibson/) for more details.

For Habitat, you need to download scenes separately. For more details, please
see [here](https://github.com/facebookresearch/habitat-lab#task-datasets).
Below, we explain how to use Replica scenes.

### How to Use Replica Scenes
* [Download Replica scenes](https://github.com/facebookresearch/Replica-Dataset).
When you run `./download.sh /path/to/replica_v1`, this path will have to be
set in the yaml file under `DATASET.SCENES_DIR`.
See for instance `example/habitat_dqn/pointnav_apartment-0.yaml`.
* Scene details, such as the agent's initial position and orientation, are defined
in `replica-start.json`.
To change the agent's initial position, you can sample one with `NavRLEnv._env._sim.sample_navigable_point()`.


In both Habitat and iGibson, The scene is defined in the yaml file of the task.
