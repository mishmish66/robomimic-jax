# robomimic on MuJoCo Warp

> **A slop fork of [robosuite](https://github.com/ARISE-Initiative/robosuite) and
> [robomimic](https://github.com/ARISE-Initiative/robomimic):** an AI-written port, not affiliated with or
> endorsed by their authors. Credit for the tasks, robots, and demonstrations is theirs.

robomimic's five tasks (Lift, Can, Square, Transport, Tool Hang) and MimicGen's 26 task variants (Stack, Stack
Three, Square, Threading, Coffee, Three Piece Assembly, Hammer Cleanup, Mug Cleanup, Kitchen, Nut Assembly, Pick
Place, Coffee Preparation, at their D0 to D2 initial-state distributions) as
[jax_pomdps](https://github.com/mishmish66/jax-pomdps) POMDPs on MuJoCo Warp, with the human demonstrations of
every robomimic v1.5 ph and mh dataset, and MimicGen's core datasets of every variant, transferred into them.

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

The robosuite and robomimic parts, and the robomimic datasets, stay under their MIT licenses, the MimicGen parts
under the NVIDIA Source Code License (research or evaluation use only), and MimicGen's datasets under CC-BY 4.0;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Everything new is under the
[VibeCoded AI-Slop License v1.0](LICENSE). Cite Mandlekar et al., "What Matters in Learning from Offline Human
Demonstrations for Robot Manipulation", CoRL 2021, and, for the MimicGen tasks and data, Mandlekar et al.,
"MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations", CoRL 2023.
