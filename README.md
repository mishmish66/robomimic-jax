# robomimic on MuJoCo Warp

> **This is a slop fork of [robosuite](https://github.com/ARISE-Initiative/robosuite) and
> [robomimic](https://github.com/ARISE-Initiative/robomimic).** It is an AI-written port of their manipulation
> tasks, controllers, models, and dataset tools to MuJoCo Warp, made by someone who wrote neither. It is not
> affiliated with or endorsed by the robosuite or robomimic authors, and credit for the tasks, the robots, and
> the demonstrations is theirs. See [License](#license).

The robomimic manipulation tasks (Lift, Can, Square, Transport, Tool Hang) as
[jax_pomdps](https://github.com/mishmish66/jax-pomdps) POMDPs simulated by MuJoCo Warp, the
robomimic demonstrations transferred into them, and tools for the robomimic datasets.

```python
import jax, jax_pomdps as pomdps

env = pomdps.make("robomimic.env:robomimic/lift", max_worlds=1024)
keys = jax.random.split(jax.random.key(0), 1024)
states = jax.jit(jax.vmap(env.reset))(keys)
states = jax.jit(jax.vmap(env.step))(keys, states, jax.vmap(env.action_space.sample)(keys))
obs = jax.jit(jax.vmap(env.observe))(keys, states, None)
```

- **Physics** runs in MuJoCo Warp. Robosuite's OSC_POSE arm controller and Panda gripper, the observations,
  rewards, and success checks run in JAX; the controller, observations, rewards, and success match robosuite
  v1.5, which recorded the v1.5 datasets.
- **Demonstrations** replay in the model they were recorded with:
  `RobomimicPOMDP(task, model_xml=f["data/demo_0"].attrs["model_file"])`, then `state_from_flat(states[t])`.
- **Cameras**: `camera_names=["agentview", ...]` adds uint8 images (and depth with `camera_depths=True`)
  rendered by the MuJoCo Warp ray tracer, shaded to approximate robosuite's OpenGL images.
- **Simulation** uses MuJoCo Playground's simulation settings: a 5 ms timestep with implicitfast integration, no
  kinematics refresh between substeps, and no collisions of the arm links (the grippers still collide), with
  the robosuite models' contact solver settings.
- [Benchmarks](benchmarks/README.md) compare throughput and fidelity with the original robosuite simulation.
- **Docs**: [docs/tasks.md](docs/tasks.md) describes the tasks and [docs/data.md](docs/data.md) the data. The
  pdoc site combines them with the API reference: `JAX_PLATFORMS=cpu uv run --with pdoc python docs/build.py
  --out site`, published to GitHub Pages by `.github/workflows/docs.yml`.

## Layout

```
src/robomimic/
  env/       the POMDP: pomdp.py, the arm controller (control.py), tasks (tasks.py), rigid-body math and contact
             tests (geometry.py), cameras (render.py), and assets/ (models, meshes, textures, task parameters,
             and initial states, exported from robosuite v1.5)
  data/      registry.py (released datasets and downloading), hdf5.py (filter keys, metadata, packing demos),
             transfer.py (transfer of demos into the POMDP), observations.py (obs and next_obs of states)
  scripts/   command-line tools, run as `python -m robomimic.scripts.<name>`
datasets/warp/   the transferred demonstrations
benchmarks/      comparison with the original robosuite simulation
docs/            the pdoc site: tasks.md, data.md, index.md, build.py, and images.py (the task images)
tests/
```

## Setup

```sh
uv sync
uv run python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types raw low_dim
uv run pytest tests
```

Datasets download to `$ROBOMIMIC_DATA` (default `~/data/robomimic_v15`). Tests on the released datasets look for
`$ROBOMIMIC_DATA/<task>/<ph|mh>/demo_v15.hdf5` and `low_dim_v15.hdf5` and skip tasks without them.

## Transferred demonstrations

`datasets/warp/` holds every demonstration of the robomimic v1.5 datasets `lift_ph`, `can_ph`, `square_mh`,
`transport_ph`, and `tool_hang_ph`, re-simulated in this POMDP: the same demos, lengths, and filter keys
(train/valid splits, operator subsets) as the released datasets, and every demo successful in MuJoCo Warp. They
were made by `robomimic.scripts.transfer_demos`, which tracks each recorded trajectory by receding-horizon
sampling of perturbations of its actions; the [benchmarks](benchmarks/README.md) report how closely they follow
the recorded ones.

The files are packed: per-step states, actions, rewards, and dones, the filter keys, and each distinct model
once, without observations. **Observations are not stored; build them** into robomimic low_dim datasets, with
`obs` and `next_obs` for robomimic's training code:

```sh
uv run python -m robomimic.scripts.make_datasets                      # all five, into ~/data/robomimic_warp
uv run python -m robomimic.scripts.make_datasets --tasks lift_ph --output ~/data/robomimic_warp
```

This writes `<output>/<task>/<type>/low_dim_v15.hdf5`, with the same observation keys and shapes as the released
low_dim datasets. `obs[t]` is the POMDP's observation of recorded state `t` and `next_obs[t]` that of state `t + 1`;
the state after a demo's last action is not recorded, so its `next_obs` observes the simulated outcome of that
action. `robomimic.scripts.add_observations` does the same for any single hdf5 dataset of recorded states.

To render a demo's replay:

```sh
uv run --extra video python -m robomimic.scripts.replay_demos --dataset <hdf5> --demos 0 1 --output_dir videos --show_dataset
```

## License

The robosuite and robomimic parts stay under their own MIT licenses: the tasks, controllers, observations,
rewards, success checks, models, meshes, textures, and initial states (robosuite, © 2022 Stanford Vision and
Learning Lab and UT Robot Perception and Learning Lab), the dataset tools (robomimic, © 2021 Stanford Vision and
Learning Lab), and the demonstrations (the robomimic datasets, MIT). If you use the demonstrations, cite
Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation", CoRL 2021.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists file by file what comes from where, with the full
license texts.

Everything new is under the [VibeCoded AI-Slop License v1.0](LICENSE).
