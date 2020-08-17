from setuptools import setup, find_packages
from codecs import open
from os import path
import sys


from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

from mushroom_rl import __version__

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))
if sys.version_info < (3, 7):
    requires_list.append('zipfile37')

extras = {
    'gym': ['gym'],
    'atari': ['atari_py~=0.2.0', 'Pillow', 'opencv-python'],
    'box2d': ['box2d-py~=2.3.5'],
    'bullet': ['pybullet'],
    'mujoco': ['mujoco_py'],
    'plots': ['pyqtgraph']
}

all_deps = []
for group_name in extras:
    if group_name not in ['mujoco', 'plots']:
        all_deps += extras[group_name]
extras['all'] = all_deps

long_description = 'Mushroom is a Python Reinforcement Learning (RL) library' \
                   ' whose modularity allows to easily use well-known Python' \
                   ' libraries for tensor computation (e.g. PyTorch, Tensorflow)' \
                   ' and RL benchmarks (e.g. OpenAI Gym, PyBullet, Deepmind' \
                   ' Control Suite). It allows to perform RL experiments in a' \
                   ' simple way providing classical RL algorithms' \
                   ' (e.g. Q-Learning, SARSA, FQI), and deep RL algorithms' \
                   ' (e.g. DQN, DDPG, SAC, TD3, TRPO, PPO). Full documentation' \
                   ' available at http://mushroomrl.readthedocs.io/en/latest/.'


ext_modules = [Extension("muscle_simulation_stepupdate",
                        ["mushroom_rl/environments/mujoco_envs/humanoid_gait/"
                         "_external_simulation/muscle_simulation_stepupdate.pyx"],
                         include_dirs=[numpy.get_include()])]

setup(
    name='mushroom-rl',
    version=__version__,
    description='A Python toolkit for Reinforcement Learning experiments.',
    long_description=long_description,
    url='https://github.com/MushroomRL/mushroom-rl',
    author="Carlo D'Eramo",
    author_email='carlo.deramo@gmail.com',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('mushroom_rl')],
    zip_safe=False,
    install_requires=requires_list,
    extras_require=extras,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                ],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules)
)
