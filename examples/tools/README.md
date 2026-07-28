# Tools

Demonstrations of the machinery *around* the RL loop, rather than of a particular algorithm or environment.
The algorithm each script happens to use is incidental — it is there to produce something to log, plot or
store.

| Script                                                           | Shows                                                                                                                                    |
|------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| [`wandb_logging.py`](wandb_logging.py)                           | the `Logger`: console output, metrics on disk, Weights & Biases logging and video recording, driven from a SAC run on Pendulum           |
| [`plotting_and_normalization.py`](plotting_and_normalization.py) | the live dataset plotting callback and the min-max state preprocessor, on an LQR run                                                     |
| [`list_backend.py`](list_backend.py)                             | the `'list'` dataset backend, which MushroomRL selects automatically when the horizon is infinite and the dataset cannot be preallocated |
| [`gridworld_viewer.py`](gridworld_viewer.py)                     | the viewer of every finite MDP, opened in turn under a random policy, so that a rendering change can be eyeballed on all of them at once |

## Weights & Biases

`wandb_logging.py` needs the optional dependency:

```bash
pip install mushroom_rl[wandb]
```

Without it, and without a set of init arguments, every wandb call is a no-op and the script still runs —
logging to the console and to disk as usual.
