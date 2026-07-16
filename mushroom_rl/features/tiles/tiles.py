import numpy as np

from .abstract_tiles import AbstractTiles


class Tiles(AbstractTiles):
    """
    Class implementing rectangular tiling. For each point in the state space,
    this class can be used to compute the index of the corresponding tile.

    """
    def __init__(self, x_range, n_tiles, dimensions=None):
        """
        Constructor.

        Args:
            x_range (list): list of two-elements lists specifying the range of each input variable. It describes
                the whole input, also when ``dimensions`` selects a subspace of it;
            n_tiles (list): list of the number of tiles to be used for each selected dimension, in the same order
                they are declared in ``dimensions``;
            dimensions (list, None): list of the dimensions of the input to be considered by the tiling. If None,
                every dimension of the input is used.

        """
        if isinstance(x_range[0], list):
            self._range = x_range
        else:
            self._range = [x_range]

        super().__init__(dimensions)

        if self._dim is not None:
            assert max(self._dim) < len(self._range)

            self._range = [self._range[d] for d in self._dim]

        if isinstance(n_tiles, list):
            assert len(n_tiles) == len(self._range)

            self._n_tiles = n_tiles
        else:
            self._n_tiles = [n_tiles] * len(self._range)

        self._size = 1

        for s in self._n_tiles:
            self._size *= s

        self._add_save_attr(
            _range='primitive',
            _n_tiles='primitive',
            _size='primitive'
        )

    def __repr__(self):
        name = f'Tiles(n_tiles={self._n_tiles}, range={np.round(self._range, 3).tolist()}'
        if self._dim is not None:
            name += f', dimensions={self._dim}'

        return name + ')'

    @staticmethod
    def generate(n_tilings, n_tiles, low, high, dimensions=None, uniform=False):
        """
        Factory method to build ``n_tilings`` tilings of ``n_tiles`` tiles with
        a range between ``low`` and ``high`` for each dimension.

        Args:
            n_tilings (int): number of tilings, or -1 to compute the number automatically;
            n_tiles (list): number of tiles for each tiling, for each selected dimension, in the same order they
                are declared in ``dimensions``;
            low (Array): lowest value for each dimension of the whole input;
            high (Array): highest value for each dimension of the whole input;
            dimensions (list, None): list of the dimensions of the input to be considered by the tilings. If None,
                every dimension of the input is used;
            uniform (bool, False): if True the displacement for each tiling will be w/n_tilings, where w is the
                tile width. Otherwise, the displacement will be k*w/n_tilings, where k=2i+1, where i is the
                dimension index.

        Returns:
            The list of the generated tiles.

        """
        low, high = AbstractTiles._to_numpy(low, high)

        assert len(low) == len(high)
        assert n_tilings > 0 or n_tilings == -1

        if dimensions is None:
            low_sub, high_sub = low, high
        else:
            low_sub, high_sub = low[dimensions], high[dimensions]

        assert len(n_tiles) == len(low_sub)

        if n_tilings == -1:
            n = np.max(n_tiles)
            d = len(low_sub)
            m = np.max([np.ceil(np.log(4 * d) / np.log(2)),
                        np.ceil(np.log(n) / np.log(2))])
            n_tilings = int(m**2)

        # Min, max coord., side length of the state-space
        L = high_sub - low_sub

        # Unit shift displacement vector
        shift = 1 if uniform else 2 * np.arange(len(low_sub)) + 1

        # Length of the sides of the tiles
        be = (n_tilings - 1) / n_tilings
        tile_side = L / (np.array(n_tiles) - be)

        # Generate the list of tilings
        tilings = list()

        for i in range(n_tilings):
            # Shift vector
            v = (i * shift) % n_tilings

            # Min, max of the coordinates of the i-th tiling
            x_min_sub = low_sub + (-n_tilings + 1 + v) / n_tilings * tile_side
            x_max_sub = x_min_sub + tile_side * n_tiles

            if dimensions is None:
                x_min, x_max = x_min_sub, x_max_sub
            else:
                x_min, x_max = low.astype(float), high.astype(float)
                x_min[dimensions] = x_min_sub
                x_max[dimensions] = x_max_sub

            # Rearrange x_min, x_max and append new tiling to the list
            x_range = [[x, y] for x, y in zip(x_min, x_max)]
            tilings.append(Tiles(x_range, n_tiles, dimensions))

        return tilings

    @property
    def size(self):
        return self._size

    def _compute(self, x):
        multiplier = 1
        tile_index = np.zeros(x.shape[0], dtype=int)
        inside = np.ones(x.shape[0], dtype=bool)

        for i, (r, N) in enumerate(zip(self._range, self._n_tiles)):
            inside &= (r[0] <= x[:, i]) & (x[:, i] < r[1])

            width = r[1] - r[0]
            component_index = np.floor(N * (x[:, i] - r[0]) / width).astype(int)
            tile_index += component_index * multiplier
            multiplier *= N

        return np.where(inside, tile_index, -1)
