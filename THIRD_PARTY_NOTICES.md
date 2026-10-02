# Third-party notices

This repository is a slop fork: an AI-written port of robomimic's and robosuite's manipulation tasks and
dataset tools to MuJoCo Warp. This file lists, file by file, where each part comes from, who made it, and under
which license, followed by the full license texts. The repository's own `LICENSE` covers only what is new;
everything below stays under its original license.

## Where each part comes from

### robosuite (MIT)

robosuite, © 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab, MIT license (full
text below). <https://github.com/ARISE-Initiative/robosuite>. The port follows robosuite v1.5.

| file | ported from robosuite |
|---|---|
| `src/robomimic/env/control.py` | the OSC_POSE arm controller (`controllers/parts/arm/osc.py`, `utils/control_utils.py`) and the Panda gripper controller (`controllers/parts/gripper/simple_grip.py`) |
| `src/robomimic/env/tasks.py` | the observations, rewards, and success checks of `environments/manipulation/lift.py`, `pick_place.py`, `nut_assembly.py`, `tool_hang.py`, `two_arm_transport.py` |
| `src/robomimic/env/pomdp.py` | the robots' proprioceptive observations (`robots/robot.py`) and the order of control and simulation in a step (`environments/base.py`) |
| `src/robomimic/env/geometry.py` | the rotation conventions of `utils/transform_utils.py` |
| `src/robomimic/env/render.py` | robosuite's camera names and image conventions |
| `src/robomimic/env/assets/models.zip` | the task models exported from robosuite's environments, and the meshes and textures of `models/assets/`, copied; robosuite's license is in the archive |
| `src/robomimic/env/assets/tasks.json`, `resets.npz` | controller and task parameters, and initial states sampled from robosuite's reset distributions, exported from robosuite's environments |
| `benchmarks/robosuite_baseline.py` | runs robosuite itself, installed when the script runs |
| `tests/robosuite_shaped_rewards.py` | runs robosuite itself, installed when the script runs |

Controllers, observations, rewards, success checks, scenes, and initial states in these files are robosuite's,
re-expressed in JAX. robosuite's own license notes that it includes a partial implementation of DeepMind's
MuJoCo, under the Apache License 2.0.

### robomimic (MIT)

robomimic, © 2021 Stanford Vision and Learning Lab, MIT license (full text below).
<https://github.com/ARISE-Initiative/robomimic>. This repository's history starts as a fork of robomimic v0.5.

| file | from robomimic |
|---|---|
| `src/robomimic/data/registry.py` | the dataset registry and Hugging Face download (`robomimic/__init__.py`, `utils/file_utils.py`) |
| `src/robomimic/data/hdf5.py` | the filter-key and metadata helpers (`utils/file_utils.py`); packing demos into npz files is new |
| `src/robomimic/data/observations.py` | the layout of `obs` and `next_obs` written by `scripts/dataset_states_to_obs.py` |
| `src/robomimic/scripts/download_datasets.py`, `get_dataset_info.py`, `split_train_val.py`, `filter_dataset_size.py` | robomimic's scripts of the same names |

### The robomimic datasets (MIT)

The robomimic v1.5 datasets, released by the robomimic authors under the MIT license
(<https://huggingface.co/datasets/robomimic/robomimic_datasets>), collected by human operators on the RoboTurk
platform. `datasets/warp/*.npz` are derived from their `lift/ph`, `can/ph`, `square/mh`, `transport/ph`, and
`tool_hang/ph` datasets: the same demos, models, and filter keys, with each demo re-simulated in this
repository's simulation. Cite their paper if you use them:

> Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio
> Savarese, Yuke Zhu, Roberto Martín-Martín. "What Matters in Learning from Offline Human Demonstrations for
> Robot Manipulation." Conference on Robot Learning (CoRL), 2021. <https://arxiv.org/abs/2108.03298>

### MuJoCo, MJX, and MuJoCo Warp (Apache 2.0)

MuJoCo, MJX, and MuJoCo Warp, © DeepMind Technologies Limited, Apache License 2.0.
<https://github.com/google-deepmind/mujoco>, <https://github.com/google-deepmind/mujoco_warp>. They are
dependencies, not copied. `src/robomimic/env/geometry.py` calls MJX's convex collision functions, and
`src/robomimic/env/render.py` imitates the shading of MuJoCo's OpenGL renderer.

### MuJoCo Playground (Apache 2.0)

MuJoCo Playground, © DeepMind Technologies Limited, Apache License 2.0.
<https://github.com/google-deepmind/mujoco_playground>. No code is copied; the simulation settings in
`src/robomimic/env/pomdp.py` (the 5 ms timestep, implicitfast integration, and colliding only the grippers of
the arms) follow its Franka Emika Panda tasks and MuJoCo Menagerie's `mjx_panda.xml`.

## License texts

### robosuite

```
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This software includes the partial implementation of Deepmind Mujoco https://github.com/deepmind/mujoco.
Deepmind Mujoco is licensed under the Apache License, Version 2.0 (the "License");
you may not use the files except in compliance with the License.

You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
```

### robomimic and the robomimic datasets

```
MIT License

Copyright (c) 2021 Stanford Vision and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Apache License 2.0

MuJoCo, MJX, MuJoCo Warp, and MuJoCo Playground: <https://www.apache.org/licenses/LICENSE-2.0>.
