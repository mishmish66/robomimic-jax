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
platform. `datasets/warp/lift_*.npz`, `can_*.npz`, `square_*.npz`, `transport_*.npz`, and `tool_hang_ph.npz`
are derived from their ph and mh datasets: the same demos, models, and filter keys, with each demo re-simulated in this
repository's simulation. Cite their paper:

> Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio
> Savarese, Yuke Zhu, Roberto Martín-Martín. "What Matters in Learning from Offline Human Demonstrations for
> Robot Manipulation." Conference on Robot Learning (CoRL), 2021. <https://arxiv.org/abs/2108.03298>

### MimicGen (NVIDIA Source Code License)

MimicGen, © 2024 NVIDIA CORPORATION & AFFILIATES, NVIDIA Source Code License (full text below), which permits use
for research or evaluation purposes only. <https://github.com/NVlabs/mimicgen>. The port follows MimicGen at commit
`72bd767`, on robosuite v1.4 at commit `b9d8d3d`, which its installation guide pins.

| file | ported from MimicGen |
|---|---|
| `src/robomimic/env/tasks.py` | the observations and success checks of the environments of `mimicgen/envs/robosuite/`: `stack.py` (`Stack_D0`, `Stack_D1`, `StackThree_D0`, `StackThree_D1`), `nut_assembly.py` (`Square_D0` to `_D2`, `NutAssembly_D0`), `threading.py` (`Threading_D0` to `_D2`), `coffee.py` (`Coffee_D0` to `_D2`, `CoffeePreparation_D0`, `_D1`), `three_piece_assembly.py` (`ThreePieceAssembly_D0` to `_D2`), `hammer_cleanup.py` (`HammerCleanup_D0`, `_D1`), `mug_cleanup.py` (`MugCleanup_D0`, `_D1`), `kitchen.py` (`Kitchen_D0`, `_D1`), and `pick_place.py` (`PickPlace_D0`) |
| `src/robomimic/env/control.py` | robosuite v1.4's gripper timing and proprioceptive observations, as MimicGen's datasets record them |
| `src/robomimic/env/assets/models.zip` | the models `models/<task>.xml` of those tasks, exported from MimicGen's environments, and the meshes and textures of `mimicgen/models/robosuite/assets/`, copied; MimicGen's license is in the archive as `LICENSE_MIMICGEN` |
| `src/robomimic/env/assets/tasks.json`, `resets.npz` | the controller and task parameters of those tasks, and initial states sampled from their reset distributions, exported from MimicGen's environments |
| `tests/mimicgen_shaped_rewards.py` | runs MimicGen and robosuite v1.4 themselves, installed when the script runs |

MimicGen's datasets are released under CC-BY 4.0 (<https://huggingface.co/datasets/amandlek/mimicgen_datasets>).
The `datasets/warp/` files named after MimicGen's tasks (`stack_d0.npz` to `coffee_preparation_d1.npz`) are derived
from their `core` datasets of those tasks: the same demos and models, each re-simulated in this repository's
simulation.
Cite their paper:

> Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox.
> "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations." Conference on Robot
> Learning (CoRL), 2023. <https://arxiv.org/abs/2310.17596>

### robosuite-task-zoo (MIT)

robosuite-task-zoo, © 2021 ARISE Initiative, MIT license (full text below).
<https://github.com/ARISE-Initiative/robosuite-task-zoo>. MimicGen's Kitchen and HammerCleanup tasks build on its
`KitchenEnv` and `HammerPlaceEnv`, at commit `74eab7f`.

| file | from robosuite-task-zoo |
|---|---|
| `src/robomimic/env/tasks.py` | the observations and success checks of `environments/manipulation/kitchen.py` and `hammer_place.py` that MimicGen keeps |
| `src/robomimic/env/assets/models.zip` | the mesh `task_zoo/models/kitchen/burnerplate.stl`, copied; robosuite-task-zoo's license is in the archive as `LICENSE_ROBOSUITE_TASK_ZOO` |

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

### robosuite-task-zoo

```
MIT License

Copyright (c) 2021 ARISE Initiative

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

### MimicGen

```
NVIDIA License

1. Definitions

“Licensor” means any person or entity that distributes its Work.
“Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.

2. License Grant

2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.

3. Limitations

3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.

3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.

3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.

3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.

3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.

3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. 

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

### Apache License 2.0

MuJoCo, MJX, MuJoCo Warp, and MuJoCo Playground: <https://www.apache.org/licenses/LICENSE-2.0>.
