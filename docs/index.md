> **This is a slop fork of [robosuite](https://github.com/ARISE-Initiative/robosuite) and
> [robomimic](https://github.com/ARISE-Initiative/robomimic).** It is an AI-written port of their manipulation
> tasks, controllers, models, and dataset tools to MuJoCo Warp, made by someone who wrote neither. It is not
> affiliated with or endorsed by the robosuite or robomimic authors, and credit for the tasks, the robots, and
> the demonstrations is theirs. The robosuite and robomimic parts stay under their MIT licenses
> (`THIRD_PARTY_NOTICES.md`); only what is new is under the VibeCoded AI-Slop License (`LICENSE`).

robomimic's five simulation tasks, Lift, Can, Square, Transport, and Tool Hang, and MimicGen's 26 task variants,
as [jax_pomdps](https://github.com/mishmish66/jax-pomdps) POMDPs simulated by MuJoCo Warp, with camera images from
this package's own renderer. It ships the human demonstrations of every robomimic v1.5 ph and mh dataset and the
26,000 demonstrations of MimicGen's core datasets, re-simulated here so that every one succeeds.

This is no longer a faithful re-implementation of robomimic. It adds MimicGen's tasks and data, simulates
everything with different physics, re-simulates every demonstration, and renders camera images of its own that do
not look like robomimic's. Use it as a set of tasks and datasets in its own right; results on it are not results on
robomimic.

```python
import jax
import jax_pomdps as pomdps

env = pomdps.make("robomimic.env:lift", max_worlds=1024)
keys = jax.random.split(jax.random.key(0), 1024)
states = jax.jit(jax.vmap(env.reset))(keys)
states = jax.jit(jax.vmap(env.step))(keys, states, jax.vmap(env.action_space.sample)(keys))
obs = jax.jit(jax.vmap(env.observe))(keys, states, None)
```

- `robomimic.env`: the tasks, their observations, actions, rewards, and success conditions.
- `robomimic.data`: the transferred demonstrations, building robomimic hdf5 datasets from them, and the
  released datasets.
- `robomimic.scripts`: command-line tools, run as `python -m robomimic.scripts.<name>`.

## What differs from robomimic and robosuite

The controller, observations, rewards, and success checks follow robosuite v1.5, which recorded the v1.5
datasets, and the MimicGen tasks follow robosuite v1.4, which recorded MimicGen's; the scenes are their own models. The physics differ: MuJoCo Warp simulates the tasks with
MuJoCo Playground's settings, a 5 ms timestep with implicitfast integration, no kinematics refresh between
substeps, and no collisions of the arm links (the grippers still collide), with the robosuite models' contact
solver settings. Recorded robosuite actions therefore do not reproduce their recorded trajectories here, which
is why the demonstrations are transferred rather than replayed: each is re-simulated from its first state with
actions searched to keep it on its recorded trajectory, and the few the search cannot bring to success are
re-executed from their recorded gripper targets or, for two MimicGen demos, generated anew (see Data). Camera
images come from the MuJoCo Warp ray tracer with shadows, antialiasing, and a camera's defocus, vignetting, white
balance, and noise; they do not imitate robosuite's OpenGL images, so the image observations differ from robomimic's
image datasets (see Tasks).

## Credits and licenses

- **robosuite**, © 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab, MIT: the
  tasks, controllers, observations, rewards, success checks, models, meshes, textures, and initial-state
  distributions.
- **robomimic**, © 2021 Stanford Vision and Learning Lab, MIT: the dataset tools, and the history this
  repository forks.
- **The robomimic datasets**, MIT: the demonstrations. Cite Mandlekar et al., "What Matters in Learning from
  Offline Human Demonstrations for Robot Manipulation", CoRL 2021.
- **MimicGen**, © 2024 NVIDIA CORPORATION & AFFILIATES, NVIDIA Source Code License (research or evaluation use
  only): the MimicGen tasks, their models, and their meshes. Its datasets are CC-BY 4.0. Cite
  Mandlekar et al., "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations",
  CoRL 2023.
- **robosuite-task-zoo**, © 2021 ARISE Initiative, MIT: the environments that MimicGen's Kitchen and HammerCleanup
  build on, and the stove's burner mesh.
- **MuJoCo, MJX, MuJoCo Warp, and MuJoCo Playground**, © DeepMind Technologies Limited, Apache 2.0:
  dependencies, and the simulation settings.
- **Everything else** is under the VibeCoded AI-Slop License v1.0, © 2026 Misha Lvovsky.

`THIRD_PARTY_NOTICES.md` lists file by file what comes from where, with the full license texts.
