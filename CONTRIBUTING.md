Contributing to MushroomRL
==========================
We strongly encourage researchers to provide us feedback and contributing
to MushroomRL. You can contribute in the following ways:
* providing bug reports;
* implementing new algorithms, approximators, environments, and so on;
* proposing improvements to the library, or documentation.

How to report bugs
------------------
Please use GitHub Issues and use the "bug" tag to label it. It is desirable if you can provide a minimal Python script
where the bug occurs. If the bug is confirmed, you can also provide a pull request to fix it, or wait for the maintainers to
resolve the issue.

Implementing new algorithms
---------------------------
Although any contribution is welcome, we will only accept high-quality code that will reflect the key ideas
of modularity and flexibility of MushroomRL. Please keep this in mind before submitting
a pull request.

### Algorithms
Every algorithm in MushroomRL must extend the ``Agent`` interface. If the algorithm belongs to a class of algorithms already
implemented in MushroomRL, then we expect the author to extend the appropriate interface. If new policies or new parameter types
are needed, please extend the corresponding base classes. Using other MushroomRL modules (e.g. ``Regressor``, ``Features``) is
highly encouraged. Together with the algorithm, you must provide an example and a test case. Every algorithm matching our
standard of code and scientific quality and providing the example and the test, will be merged in MushroomRL. It is encouraged
to include the citation to the paper describing the algorithm in the docstring of the class.

### Function approximators
MushroomRL supports the use of third-party function approximators (e.g. Scikit-learn, Keras),
or customized ones, as long as they implement the ``fit`` and ``predict`` methods.

### Environments
The environment template of MushroomRL follows the style of the one introduced in OpenAI Gym.
Customized environment can be added provided that this exact style is used, and extending
the ``Environment`` interface. Customized functions for visualization, i.e., ``render``, and
others can be easily added.

### Examples
MushroomRL provides several examples showing the expected instructions sequence that an
experiment should have. This structure is not mandatory, but recommended. Although all
the examples share some similarities, there are some differences depending on the
algorithm and environments at hand, and also on the specific needs of the user. It is
desirable to propose an example whenever a new algorithm or environment is proposed in
a pull request. The choice of the type of experiment is up to the contributor, but it should
be meaningful to show the performance and properties of the proposed algorithm/environment.

### Benchmarks
Instructions for adding a new benchmark are provided in the [MushroomRL Benchmarking Suite](
https://github.com/MushroomRL/mushroom-rl-benchmark).

Proposing improvements
----------------------
If you find some lack of features, please open a GitHub issue and use the "enhancement" tag to label it. However, before
opening a pull request, we suggest to discuss it with the maintainers of the library, that will be glad of providing
suggestions and help in case the proposed improvement is particularly interesting.
