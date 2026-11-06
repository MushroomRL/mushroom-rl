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
where the bug occurs. If the bug is confirmed, you can also provide a pull request to fix it, or wait for the 
maintainers to resolve the issue.

Extending MushroomRL
--------------------
Although any contribution is welcome, we will only accept high-quality code that will reflect the key ideas
of modularity and flexibility of MushroomRL. Please keep this in mind before submitting a pull request.

### Algorithms
Every algorithm in MushroomRL must extend the ``Agent`` interface. If the algorithm belongs to a class of algorithms 
already implemented in MushroomRL, then we expect the author to extend the appropriate interface. If new policies or new 
parameter types are needed, please extend the corresponding base classes. Using other MushroomRL modules (e.g. 
``Approximator``, ``Features``) is highly encouraged. Together with the algorithm, you must provide an example and a 
test case. 
Every algorithm matching our standard of code and scientific quality and providing the example and the test, will be 
merged in MushroomRL. 
It is encouraged to include the citation to the paper describing the algorithm in the docstring of the class.

### Function approximators
Any function approximator can be used, provided it implements the ``Approximator`` interface;
scikit-learn models are supported through ``SklearnApproximator``.

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

### Tests
Every contribution must come with tests. Run them with ``pytest --cov=mushroom_rl --cov-branch``: a pull request
must not lower line or branch coverage. The layout of ``tests/`` is mandatory:
* ``tests/`` mirrors the package structure: every folder corresponds to a ``mushroom_rl`` subpackage, apart from
  ``tests/helper/`` (shared helpers) and the ``data/`` folders (recorded data);
* add new tests to the existing test file of the module or component you change; create a new test file only for a
  new module or component, in the mirrored folder and named after it. Pull requests adding ad-hoc test files,
  or splitting or merging existing ones, will not be accepted.

Please also follow the conventions of the existing tests:
* reuse the helpers in ``tests/helper/`` and parametrize over backends and devices (``backend_devices()``,
  ``devices()``) instead of duplicating tests;
* no module-level variables: inputs, seeds and expected values are defined inside each test function;
* expected values are exact numbers recorded once with a fixed seed, never range checks; recorded arrays of four
  or more values are stored as ``.npy`` files in the ``data/`` folder next to the test file;
* tests needing an optional package skip with ``pytest.importorskip`` on that package;
* short test names, several related asserts per test, and no tests of trivial behaviour that add no coverage;
* a test for a bug fix must fail without the fix.

### Benchmarks
Instructions for adding a new benchmark are provided in the [MushroomRL Benchmarking Suite](
https://github.com/MushroomRL/mushroom-rl-benchmark/blob/master/CONTRIBUTING.md).

Proposing improvements
----------------------
If you find some lack of features, please open a GitHub issue and use the "enhancement" tag to label it. However, before
opening a pull request, we suggest to discuss it with the maintainers of the library, that will be glad of providing
suggestions and help in case the proposed improvement is particularly interesting.

AI-generated pull requests
--------------------------
Everyone is welcome to contribute to MushroomRL. However, maintaining the library becomes very difficult when we are
flooded with pull requests making minor, irrelevant changes, and reviewing code and tests that do not follow the
style of the library often takes us more time than fixing the problem ourselves. For this reason, we do not accept
unsolicited AI-generated pull requests, unless they fix a major bug in the library: those proposing minor changes will
be closed without review, and users who keep submitting them will be blocked. If you have found a minor bug, please
open a GitHub issue instead: a short description of the problem is much easier for us to handle than a pull request.
