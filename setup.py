from setuptools import setup, find_packages
from codecs import open
from os import path

from mushroom import __version__

here = path.abspath(path.dirname(__file__))
name = 'mushroom'

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

extras = {
    'gym': ['gym'],
    'atari': ['atari_py~=0.2.0', 'Pillow', 'opencv-python'],
    'box2d': ['box2d-py~=2.3.5'],
    'bullet': ['pybullet'],
    'mujoco': ['mujoco_py']
}

all_deps = []
for group_name in extras:
    if group_name != 'mujoco':
        all_deps += extras[group_name]
extras['all'] = all_deps

setup(
    name=name,
    version=__version__,
    description='A Python toolkit for Reinforcement Learning experiments.',
    url='https://github.com/AIRLab-POLIMI/mushroom',
    author="Carlo D'Eramo",
    author_email='carlo.deramo@gmail.com',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith(name)],
    zip_safe=False,
    install_requires=requires_list,
    extras_require=extras
)
