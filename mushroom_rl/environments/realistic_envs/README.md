# Habitat Installation

### 1. Install Replica scenes
```
sudo apt-get install pigz
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh replica-path
```
Be sure that Replica path is the same defined in `pointnav_nomap.yaml`.
By default, this is `Replica-Dataset/replica-path`.

### 2. Install habitat-sim
```
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
# We require python>=3.6 and cmake>=3.10
conda create -n habitat python=3.6 cmake=3.14.0
conda activate habitat
pip install -r requirements.txt
python setup.py install --headless --with-cuda
```
More detailed instructions [here](https://github.com/facebookresearch/habitat-sim/blob/master/BUILD_FROM_SOURCE.md).
You can also [install with conda](https://github.com/facebookresearch/habitat-sim#installation), but be sure to install `stable` branch.

### 3. Install habitat-lab
The full install including `habitat_baselines` is required.
```
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```

### 4. Test
You can test your Habitat install by running
```
python examples/example.py --scene=/path/to/Replica-Dataset/replica-path/apartment_0/habitat/mesh_semantic.ply
```
Now you are all set!

### 5. Scene configuration
Scene details, such as the agent's initial position and orientation, are defined in `replica-start.json`.
The agent's initial position, though, depends on the random seed passed to the run, and it is read from `scene_locations.txt`.
The n-th seed reads the n-th initial position defined in the file. These positions have been randomly chosen from the set of each scene navigable points, accessible by `HabitatWrapper.env._env._sim.sample_navigable_point()`.



# iGibson Installation
Follow [the official guide](http://svl.stanford.edu/igibson/).
