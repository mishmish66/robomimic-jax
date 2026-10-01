# robomimic on MuJoCo Warp

The robomimic manipulation tasks (Lift, Can, Square, Transport, Tool Hang) as
[jax_mdps](https://github.khoury.northeastern.edu/mishmish/jax_mdps) POMDPs simulated by MuJoCo Warp, plus
the tools for the robomimic demonstration datasets.

```python
import jax, jax_mdps as mdps

env = mdps.make("robomimic.pomdp:robomimic/lift", max_worlds=1024)
keys = jax.random.split(jax.random.key(0), 1024)
states = jax.jit(jax.vmap(env.reset))(keys)
states = jax.jit(jax.vmap(env.step))(keys, states, jax.vmap(env.action_space.sample)(keys))
obs = jax.jit(jax.vmap(env.observe))(keys, states, None)
```

- **Physics** runs in MuJoCo Warp. Robosuite's OSC_POSE arm controller and Panda gripper, the observations,
  rewards, and success checks run in JAX and match robosuite v1.5 (the version that recorded the v1.5 datasets).
- **Demonstrations** replay in the model they were recorded with:
  `RobomimicPOMDP(task, model_xml=f["data/demo_0"].attrs["model_file"])`, then `state_from_flat(states[t])`.
- **Cameras**: `camera_names=["agentview", ...]` adds uint8 images (and depth with `camera_depths=True`)
  rendered by the MuJoCo Warp ray tracer, shaded to approximate robosuite's OpenGL images.
- `data/` holds each task's model, meshes and textures, controller and task parameters, and 1024 initial states,
  exported from robosuite v1.5.

## Setup

```sh
uv sync
uv run python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph --hdf5_types raw low_dim
uv run --extra video python robomimic/scripts/replay_demos.py --dataset <path to hdf5> --demos 0 1 --output_dir videos --show_dataset
uv run pytest tests
```

Datasets download to `$ROBOMIMIC_DATA` (default `~/data/robomimic_v15`). Tests on the full task datasets look for
`$ROBOMIMIC_DATA/<task>/<ph|mh>/demo_v15.hdf5` and `low_dim_v15.hdf5` and skip tasks without them.

## Transferred demonstrations

`datasets/warp/` holds the robomimic v1.5 demonstrations re-simulated in this POMDP, every demo successful and
matching the original demo for demo (`lift_ph`, `can_ph`, `square_mh`, `transport_ph`, `tool_hang_ph`). They
were made by `robomimic/scripts/transfer_demos.py`, which tracks each recorded trajectory by sampling
perturbations of its actions. Each file packs states, actions, rewards, masks, and models; to get a robomimic
low_dim dataset with `obs` and `next_obs`:

```sh
uv run python robomimic/scripts/add_observations.py --npz datasets/warp/lift_ph.npz --dataset lift_ph_warp.hdf5
```

## License

MIT (robomimic). The models, meshes and textures in `robomimic/data/models.zip` come from robosuite (MIT; its license is in the archive).
