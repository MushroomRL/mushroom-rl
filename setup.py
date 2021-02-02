from setuptools import setup, find_packages
from codecs import open
from os import path
import sys
import glob


from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

from mushroom_rl import __version__


def glob_data_files(data_package, data_type=None):
    data_type = '*' if data_type is None else data_type
    data_dir = data_package.replace(".", "/")
    data_files = [] 
    directories = glob.glob(data_dir+'/**/', recursive=True) 
    for directory in directories:
        subdir = directory[len(data_dir)+1:]
        if subdir != "":
            files = subdir  + data_type
            data_files.append(files)
    return data_files

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

long_description = 'MushroomRL is a Python Reinforcement Learning (RL) library' \
                   ' whose modularity allows to easily use well-known Python' \
                   ' libraries for tensor computation (e.g. PyTorch, Tensorflow)' \
                   ' and RL benchmarks (e.g. OpenAI Gym, PyBullet, Deepmind' \
                   ' Control Suite). It allows to perform RL experiments in a' \
                   ' simple way providing classical RL algorithms' \
                   ' (e.g. Q-Learning, SARSA, FQI), and deep RL algorithms' \
                   ' (e.g. DQN, DDPG, SAC, TD3, TRPO, PPO). Full documentation' \
                   ' available at http://mushroomrl.readthedocs.io/en/latest/.'


ext_modules = [Extension('mushroom_rl.environments.mujoco_envs.humanoid_gait.'
                         '_external_simulation.muscle_simulation_stepupdate',
                        ['mushroom_rl/environments/mujoco_envs/humanoid_gait/'
                         '_external_simulation/muscle_simulation_stepupdate.pyx'],
                         include_dirs=[numpy.get_include()])]

mujoco_data_package = 'mushroom_rl.environments.mujoco_envs.data'
extrenal_simulation_package = 'mushroom_rl.environments.mujoco_envs.humanoid_gait._external_simulation'

setup(
    name='mushroom-rl',
    version=__version__,
    description='A Python library for Reinforcement Learning experiments.',
    long_description=long_description,
    url='https://github.com/MushroomRL/mushroom-rl',
    author="Carlo D'Eramo, Davide Tateo",
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
    package_data={
        mujoco_data_package: glob_data_files(mujoco_data_package),
        extrenal_simulation_package: ["*.pyx"]},
    ext_modules=cythonize(ext_modules)
)
