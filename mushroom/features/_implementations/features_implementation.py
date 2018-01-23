class FeaturesImplementation(object):
    def __call__(self, *x):
        """
        Evaluate the feature vector in the given raw input. If more than one
        element is passed, the raw input is concatenated before computing the
        features.

        Args:
            *x (list): the raw input.

        Returns:
            The features vector computed from the raw input.

        """
        pass

    @property
    def size(self):
        """
        Returns:
             The number of elements in the features vector.

        """
        pass
