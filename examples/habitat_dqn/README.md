## Habitat Installation

1. [Install habitat-sim](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md).
You can also [install with conda](https://github.com/facebookresearch/habitat-sim#installation),
but be sure to install `stable` branch.

2. [Install habitat-lab](https://github.com/facebookresearch/habitat-lab).
The full installation including `habitat_baselines` is required.


## iGibson Installation
Follow the official guide [here](https://github.com/StanfordVL/iGibson).


## Scene Datasets
Habitat and iGibson support many realistic scenes as environment for the agent.
iGibson has also its own dataset, that can be downloaded and used right away.
Please see the [official documentation](http://svl.stanford.edu/igibson/) for
more details.

For Habitat, you need to download scenes separately. For more details, please
see [here](https://github.com/facebookresearch/habitat-lab#task-datasets).
Here, we explain how to use Replica scenes.

### Replica Scenes Installation
[Download Replica scenes](https://github.com/facebookresearch/Replica-Dataset).
When you run `./download.sh /path/to/replica_v1`, this path will have to be
passed to `HabitatNav`, together with the desired scene to use. Alternatively,
it can be directly set in the yaml file under `DATASET.SCENES_DIR`.



In both Habitat and iGibson, The scene is defined in the yaml file of the task.


## Scene configuration
Scene details, such as the agent's initial position and orientation, are defined
in `replica-start.json`.
To change the agent's initial position, you can sample one with `NavRLEnv._env._sim.sample_navigable_point()`.
