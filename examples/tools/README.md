# Tools

Demonstrations of the MushroomRL functionalities. The scripts focus on specific mechanisms rather than on a particular 
algorithm or environment.

| Script                                                               | Shows                                                                                                                                    |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| [`wandb_logging.py`](wandb_logging.py)                               | the `Logger`: console output, metrics on disk, Weights & Biases logging and video recording, driven from a SAC run on Pendulum           |
| [`multiprocess_env_recording.py`](multiprocess_env_recording.py)     | video recording of a `MultiprocessEnvironment`, where the episodes of the parallel copies end up one after the other in a single video   |
| [`monitoring_and_normalization.py`](monitoring_and_normalization.py) | the live dataset monitoring callback and the min-max state preprocessor, on an LQR run                                                   |
| [`list_backend.py`](list_backend.py)                                 | the `'list'` dataset backend, which MushroomRL selects automatically when the horizon is infinite and the dataset cannot be preallocated |
| [`gridworld_viewer.py`](gridworld_viewer.py)                         | the viewer of every finite MDP, opened in turn under a random policy, so that a rendering change can be eyeballed on all of them at once |

## Weights & Biases

`wandb_logging.py` needs the optional dependency:

```bash
pip install mushroom_rl[wandb]
```

Without it, and without a set of init arguments, every wandb call is a no-op and the script still runs — logging to the
console and to disk as usual.
