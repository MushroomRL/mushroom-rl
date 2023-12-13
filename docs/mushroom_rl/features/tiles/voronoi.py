import numpy as np


class VoronoiTiles:
    """
    Class implementing voronoi tiling. For each point in the state space,
    this class can be used to compute the index of the corresponding tile.

    """
    def __init__(self, prototypes):
        """
        Constructor.

        Args:
            prototypes (list): list of prototypes to compute the partition.

        """
        self._prototypes = prototypes

    def __call__(self, x):
        dist = np.linalg.norm(self._prototypes - x, axis=1)

        return np.argmin(dist)

    @staticmethod
    def generate(n_tilings, n_prototypes, low=None, high=None, mu=None, sigma=None):
        """
        Factory method to build ``n_tilings`` tilings of ``n_prototypes``.
        Prototypes are generated randomly sampled. If low and high are provided,
        prototypes are sampled uniformly between low and high, otherwise mu and
        sigma must be specified and prototypes are sampled from the corresponding
        Gaussian.

        Args:
            n_tilings (int): number of tilings, or -1 to compute the number
                             automatically;
            n_prototypes (list): number of prototypes for each tiling;
            low (np.ndarray, None): lowest value for each dimension, needed for
                                    uniform sampling;
            high (np.ndarray, None): highest value for each dimension, needed for
                                     uniform sampling.
            mu (np.ndarray, None): mean value for each dimension, needed for
                                   Gaussian sampling.
            sigma (np.ndarray, None): variance along each dimension, needed for
                                      Gaussian sampling.

        Returns:
            The list of the generated tiles.

        """

        assert (low is not None and high is not None and mu is None and sigma is None) \
               or (mu is not None and sigma is not None and low is None and high is None)

        uniform = low is not None

        if uniform:
            assert len(low) == len(high)
        else:
            assert len(mu) == len(sigma)

        assert n_tilings > 0 or n_tilings == -1

        if n_tilings == -1:
            d = np.size(low)  # space-dim
            m = np.max([np.ceil(np.log(4 * d) / np.log(2)),
                        np.ceil(np.log(n_prototypes) / np.log(2))])
            n_tilings = int(m**2)

        if uniform:
            low = np.array(low, dtype=float)
            high = np.array(high, dtype=float)
        else:
            mu = np.array(mu, dtype=float)
            sigma = np.array(sigma, dtype=float)

        # Generate the list of tilings
        tilings = list()

        for i in range(n_tilings):
            if uniform:
                prototypes = np.random.uniform(low, high, (n_prototypes, len(low)))
            else:
                prototypes = np.random.normal(mu, sigma, (n_prototypes, len(mu)))
            tilings.append(VoronoiTiles(prototypes))

        return tilings

    @property
    def size(self):
        return len(self._prototypes)
