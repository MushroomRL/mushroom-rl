# MushroomRL examples

Runnable scripts showing how to use MushroomRL. Every script can be launched directly from any working
directory:

```bash
python examples/algorithms/value/simple_chain_qlearning.py
```

The examples are grouped as follows:

| Folder                          | Contents                                                 |
|---------------------------------|----------------------------------------------------------|
| [`papers/`](papers)             | *Scripts reproducing experiments from a published paper* |
| [`algorithms/`](algorithms)     | *Examples of algorithm usage*                            |
| [`environments/`](environments) | *Examples of some specific environment interface*        |
| [`tools/`](tools)               | *Basic usage examples of the tools MushroomRL provides*  |

Each of those folders has its own README describing its contents.

## Shared folders

- **`data/`** — input files loaded by the examples, shared across the tree. Scripts locate this folder with
  `get_data_dir(__file__)`, so it does not matter where they are run from.
- **`logs/`** — every run writes here, whatever the depth of the script that produced it. Scripts locate it
  with `get_log_dir(__file__)`. The folder is git-ignored.

Both helpers live in `mushroom_rl.utils` and resolve the paths from the script's own location, which is why
the examples do not care about the current working directory.

## Conventions

Every example follows the same shape: a module docstring stating what it shows, an `experiment()` function
holding the run, and a `__main__` block that sets the hyperparameters and calls it. `experiment()` takes a
`seed` as its last argument, so a script can be reproduced by importing it and passing one.

Some scripts allow to pass command line arguments. Those scripts carry a `parse_args()` function next to `experiment()`.

Each script builds the environment first and the `Logger` right after, so that `log_experiment_info` can
name both the algorithm and the environment; the per-epoch metrics then go through `log_evaluation`, which
prints them and, when the logger has a results directory, writes them to disk.
