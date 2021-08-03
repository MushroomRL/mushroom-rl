This wrapper is designed for Mujoco environments from the
[`mj_envs` repository](https://github.com/vikashplus/mj_envs), such as the
[hand manipulation suite](https://sites.google.com/view/deeprl-dexterous-manipulation).
Note that `mj_envs` has many branches and it is also included in other
repositories. Each branch / repository has modified versions of the environments.
Our wrapper is compatible with any `mj_envs` environment.


## Install mujoco-py
Follow the [official instructions](https://github.com/openai/mujoco-py/).
To check if it was correctly installed on GPU, execute the following python code
```
python -c 'import mujoco_py ; print(mujoco_py.cymj)'
```
You will see something like
```
<module 'cymj' from '/path/to/mujoco_py/generated/cymj_2.0.2.13_37_linuxcpuextensionbuilder_37.so'>
```
If you see `linuxgpuextensionbuilder` then `mujoco-py` has been correctly installed
on GPU. If you see `linuxcpuextensionbuilder` see below.

### Fix mujoco-py with GPU support
Before installing with `pip`, edit `mujoco_py/builder.py` [this line](https://github.com/openai/mujoco-py/blob/d73ce6e91d096b74da2a2fcb0a4164e10db5f641/mujoco_py/builder.py#L74) and change
it from CPU to GPU.


## Install Original Environments
```
git clone --recursive https://github.com/vikashplus/mj_envs.git
cd mj_envs  
git submodule update --remote
pip install -e .
```

### Version with pixel observations
This will replace the original installation.
```
git clone git@github.com:vikashplus/latentRL.git
cd latentRL/resnet
pip install -e mj_envs/.
```

## Install MJRL
```
https://github.com/aravindr93/mjrl
git clone git@github.com:aravindr93/mjrl.git
cd mjrl
pip install -e .
```
