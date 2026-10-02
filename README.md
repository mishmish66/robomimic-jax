# robomimic on MuJoCo Warp

> **A slop fork of [robosuite](https://github.com/ARISE-Initiative/robosuite) and
> [robomimic](https://github.com/ARISE-Initiative/robomimic):** an AI-written port, not affiliated with or
> endorsed by their authors. Credit for the tasks, robots, and demonstrations is theirs.

robomimic's five tasks (Lift, Can, Square, Transport, Tool Hang) as
[jax_pomdps](https://github.com/mishmish66/jax-pomdps) POMDPs on MuJoCo Warp, with all 1200 robomimic v1.5
demonstrations transferred into them.

**[Documentation](https://mishmish66.github.io/robomimic-jax/)**: the tasks, the data, and the API.

```python
import jax, jax_pomdps as pomdps

env = pomdps.make("robomimic.env:lift", max_worlds=1024)
keys = jax.random.split(jax.random.key(0), 1024)
states = jax.jit(jax.vmap(env.reset))(keys)
states = jax.jit(jax.vmap(env.step))(keys, states, jax.vmap(env.action_space.sample)(keys))
```

## Setup

```sh
uv sync
uv run python -m robomimic.scripts.make_datasets   # robomimic low_dim hdf5s of datasets/warp, in ~/data/robomimic_warp
uv run pytest tests
```

`datasets/warp/` stores states and actions without observations; `make_datasets` adds them. Tests compare against
the released datasets in `$ROBOMIMIC_DATA` (default `~/data/robomimic_v15`; fetch them with
`robomimic.scripts.download_datasets`) and skip tasks without them.

[Benchmarks](benchmarks/README.md) compare throughput and fidelity with robosuite.

## License

The robosuite and robomimic parts, and the robomimic datasets, stay under their MIT licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Everything new is under the
[VibeCoded AI-Slop License v1.0](LICENSE). If you use the demonstrations, cite Mandlekar et al., "What Matters in
Learning from Offline Human Demonstrations for Robot Manipulation", CoRL 2021.
