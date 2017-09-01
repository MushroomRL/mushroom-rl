# Mushroom
## Reinforcement Learning python library

Mushroom is a Reinforcement Learning (RL) library that aims to be a simple, yet
powerful way to make RL experiments in a fast way. The idea behind Mushroom consists
in offering the majority of RL algorithms providing a common interface
in order to run them without excessive effort. It makes a large use of the environments
provided by [OpenAI Gym](https://gym.openai.com/) library and of the regression models
provided by [Scikit-Learn](http://scikit-learn.org/stable/) library giving also the possibility
to build and run neural networks using [Tensorflow](https://www.tensorflow.org) library.

With Mushroom you can:

* Solve value-based RL problems simply writing a single small script.
* Use all RL environments offered by OpenAI Gym and build customized environments as well.
* Exploit regression models offered by Scikit-Learn or build a customized one with Tensorflow.
* Seamlessly run experiments with CPU or GPU.

## Basic run example

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs help` - Print this help message.

## Installation

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
