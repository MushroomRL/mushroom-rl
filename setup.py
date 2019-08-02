"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
name = 'mushroom'

# Get the long description from the README file
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

# specific dependencies for modules
extras = {
    'gym': ['gym'],
    'atari': ['gym[atari]'],
    'bullet': ['pybullet'],
    'mujoco': ['mujoco_py']
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    if group_name != 'mujoco':
        all_deps += extras[group_name]
extras['all'] = all_deps

setup(
    name=name,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.1.0',

    description='A Python toolkit for Reinforcement Learning experiments.',

    # The project's main homepage.
    url='https://github.com/AIRLab-POLIMI/' + name,

    # Author details
    author="Carlo D'Eramo",
    author_email='carlo.deramo@gmail.com',

    # Choose your license
    license='',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=[package for package in find_packages()
              if package.startswith(name)],

    zip_safe=False,

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=requires_list,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,tests,all]
    extras_require=extras
)
