# Data

## The transferred demonstrations

`datasets/warp/` holds every demonstration of five robomimic v1.5 datasets, re-simulated in this repository's
simulation so that each one succeeds here:

| file | task | demos | collected by |
|---|---|---|---|
| `lift_ph.npz` | Lift | 200 | one proficient operator |
| `can_ph.npz` | Can | 200 | one proficient operator |
| `square_mh.npz` | Square | 300 | six operators of mixed proficiency, 50 demos each |
| `transport_ph.npz` | Transport | 200 | proficient operators |
| `tool_hang_ph.npz` | Tool Hang | 200 | one proficient operator |

Each file keeps its dataset's demo names, lengths, and filter keys: the `train` and `valid` splits, the
`20_percent` and `50_percent` subsets, and, for Square, the operator and quality subsets (`better`, `okay`,
`worse`, `better_operator_1`, ...). The human demonstrations were recorded in robosuite on CPU MuJoCo, whose
physics differ from this simulation's, so replaying a demo's recorded actions here drifts from its recorded
states. `robomimic.data.transfer` closes that gap: from each demo's first state, it rolls out perturbations of
the recorded actions in parallel worlds, a few steps at a time, and keeps those whose states stay closest to the
recorded ones, until the demo succeeds. The files hold the simulated states, the actions that produced them,
and the rewards and dones they earned.

How far the transferred demos' objects stray from their recorded positions, in mm (median / 95th percentile /
max over demos): the largest distance of any object at any step, and at the last step.

| task | successful | largest object deviation | final object deviation |
|---|---|---|---|
| Lift | 200/200 | 5.2 / 8.9 / 13.9 | 3.7 / 6.7 / 13.3 |
| Can | 200/200 | 48.0 / 83.8 / 148.1 | 11.0 / 25.0 / 67.8 |
| Square | 300/300 | 11.3 / 81.7 / 136.0 | 3.1 / 7.6 / 77.8 |
| Transport | 200/200 | 30.7 / 167.8 / 843.1 | 15.4 / 91.0 / 839.5 |
| Tool Hang | 200/200 | 22.5 / 81.1 / 142.2 | 12.4 / 57.3 / 99.1 |

## Building robomimic datasets

The packed files store no observations. Build them into robomimic low_dim hdf5 datasets, with `obs` and
`next_obs` for robomimic's training code:

```sh
uv run python -m robomimic.scripts.make_datasets          # all five, into ~/data/robomimic_warp/<task>/<type>/low_dim_v15.hdf5
uv run python -m robomimic.scripts.make_datasets --tasks lift_ph --output <dir>
```

The datasets have the released low_dim datasets' layout, observation keys, and shapes. `obs[t]` is the POMDP's
observation of recorded state `t`, and `next_obs[t]` that of state `t + 1`; the state after a demo's last
action is not recorded, so its last `next_obs` observes the simulated outcome of that action. Load a packed
file directly with `robomimic.data.hdf5.unpack_demos`, which writes the hdf5 layout without observations.

## Packed format

An npz of arrays, concatenated over demos in `names` order:

| key | contents |
|---|---|
| `env_args` | the dataset's environment metadata, a JSON string |
| `names`, `lengths` | demo names and their numbers of steps |
| `models`, `model_index` | each distinct robosuite model xml once, and each demo's index into them |
| `success` | whether each demo succeeds |
| `step/states` | flattened MuJoCo states `[time, qpos, qvel]` before each action, float32 |
| `step/actions`, `step/rewards`, `step/dones` | per step; actions and rewards float32 |
| `step/<annotation>` | the dataset's other per-step annotations, unchanged: `interventions`, `policy_acting`, `user_acting` |
| `mask/<key>` | the demo names of each filter key |

## The released datasets

`robomimic.data.registry` lists the released robomimic v1.5 datasets on Hugging Face, and
`robomimic.scripts.download_datasets` downloads them into `$ROBOMIMIC_DATA` (default `~/data/robomimic_v15`):

```sh
uv run python -m robomimic.scripts.download_datasets --tasks square --dataset_types mh --hdf5_types raw low_dim
```

`robomimic.scripts.transfer_demos` transfers any of them; `get_dataset_info`, `split_train_val`, and
`filter_dataset_size` inspect and split robomimic hdf5 datasets as robomimic's scripts do.
