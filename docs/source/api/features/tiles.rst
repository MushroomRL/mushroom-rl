Tiles
=====

Tilings discretize the input space and return the one-hot encoding of the tile the input falls into. The rectangular
tiles split each dimension on a regular grid; the Voronoi ones assign the input to the nearest of a set of prototypes.

.. automodule:: mushroom_rl.features.tiles.abstract_tiles
    :private-members:

.. automodule:: mushroom_rl.features.tiles.tiles
    :inherited-members:

.. automodule:: mushroom_rl.features.tiles.voronoi
    :inherited-members:
