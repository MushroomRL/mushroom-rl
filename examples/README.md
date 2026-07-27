# MushroomRL examples

Runnable scripts showing how to use MushroomRL. Every script can be launched directly from any working
directory:

```bash
python examples/algorithms/value/simple_chain_qlearning.py
```

The examples are grouped by *purpose*, not by algorithm family:

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
holding the run, a `parse_args()` function for the command line, and a `__main__` block wiring the two
together. Common options are `--seed` (omit it for a non-reproducible run) and `--no-render` (skip the
visualisation, needed on a headless machine).
