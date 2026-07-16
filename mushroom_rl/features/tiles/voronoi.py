import numpy as np

from .abstract_tiles import AbstractTiles


class VoronoiTiles(AbstractTiles):
    """
    Class implementing voronoi tiling. For each point in the state space,
    this class can be used to compute the index of the corresponding tile.

    """
    def __init__(self, prototypes, dimensions=None):
        """
        Constructor.

        Args:
            prototypes (Array): array of prototypes to compute the partition. It lies in the subspace selected by
                ``dimensions``, so its dimensionality must match the number of selected dimensions;
            dimensions (list, None): list of the dimensions of the input to be considered by the tiling. If None,
                every dimension of the input is used.

        """
        self._prototypes = self._to_numpy(prototypes)

        super().__init__(dimensions)

        if self._dim is not None:
            assert len(self._dim) == self._prototypes.shape[1]

        self._add_save_attr(_prototypes='numpy')

    def __repr__(self):
        name = f'VoronoiTiles(n_prototypes={self.size}'
        if self._dim is not None:
            name += f', dimensions={self._dim}'

        return name + ')'

    @staticmethod
    def generate(n_tilings, n_prototypes, low=None, high=None, mu=None, sigma=None, dimensions=None):
        """
        Factory method to build ``n_tilings`` tilings of ``n_prototypes``.
        Prototypes are generated randomly sampled. If low and high are provided, prototypes are sampled uniformly
        between low and high, otherwise mu and sigma must be specified and prototypes are sampled from the corresponding
        Gaussian.

        Args:
            n_tilings (int): number of tilings, or -1 to compute the number automatically;
            n_prototypes (int): number of prototypes of each tiling;
            low (Array, None): lowest value for each dimension of the whole input, needed for uniform sampling;
            high (Array, None): highest value for each dimension of the whole input, needed for uniform sampling;
            mu (Array, None): mean value for each selected dimension, needed for Gaussian sampling;
            sigma (Array, None): variance along each selected dimension, needed for Gaussian sampling;
            dimensions (list, None): list of the dimensions of the input to be considered by the tilings. If None,
                every dimension of the input is used.

        Returns:
            The list of the generated tiles.

        """
        assert (low is not None and high is not None and mu is None and sigma is None) \
               or (mu is not None and sigma is not None and low is None and high is None)

        uniform = low is not None

        if uniform:
            low, high = AbstractTiles._to_numpy(low, high)
            assert len(low) == len(high)

            if dimensions is not None:
                low, high = low[dimensions], high[dimensions]

            n_features = len(low)
        else:
            mu, sigma = AbstractTiles._to_numpy(mu, sigma)
            assert len(mu) == len(sigma)
            assert dimensions is None or len(mu) == len(dimensions)

            n_features = len(mu)

        assert n_tilings > 0 or n_tilings == -1

        if n_tilings == -1:
            d = n_features
            m = np.max([np.ceil(np.log(4 * d) / np.log(2)),
                        np.ceil(np.log(n_prototypes) / np.log(2))])
            n_tilings = int(m**2)

        # Generate the list of tilings
        tilings = list()

        for i in range(n_tilings):
            if uniform:
                prototypes = np.random.uniform(low, high, (n_prototypes, n_features))
            else:
                prototypes = np.random.normal(mu, sigma, (n_prototypes, n_features))
            tilings.append(VoronoiTiles(prototypes, dimensions))

        return tilings

    @property
    def size(self):
        return len(self._prototypes)

    def _compute(self, x):
        dist = np.linalg.norm(x[:, np.newaxis, :] - self._prototypes, axis=-1)

        return np.argmin(dist, axis=-1)
