# Data

Input files read by the examples: environment maps, precomputed matrices, and anything else a script loads
rather than generates.

They are kept here, instead of next to the script that reads them, so that the same file can be shared by
several examples and so that moving a script does not break it. Scripts resolve this folder with
`get_data_dir(__file__)` from `mushroom_rl.utils`, which locates it from the script's own position, so they
work from any working directory.

Only inputs belong here. Anything a run produces goes to `../logs/`.
