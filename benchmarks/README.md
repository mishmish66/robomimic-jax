# Benchmarks: robosuite (original robomimic) vs MuJoCo Warp

robomimic simulates its tasks with robosuite on CPU MuJoCo, one environment per process. This repo simulates
the same tasks in MuJoCo Warp, many worlds at once on a GPU. The benchmarks compare the two on:

- **Throughput**: env steps per second (one step and one observation of every world), with low-dim
  observations and with two 84x84 RGB cameras. robosuite is timed in one process and in one process per CPU
  thread; Warp at growing numbers of worlds stepped together.
- **Fidelity to the released demonstrations** (the first 16 demos of each dataset): open-loop replay of each
  demo's recorded actions from its first state, and the error of object positions and arm joints after 1 and 10
  recorded actions taken from recorded states. robosuite 1.5.1 is the version the v1.5 datasets target.
- **Transferred demonstrations**: success and tracking of every demo in `datasets/warp`.

```sh
for t in lift can square transport tool_hang; do
  uv run python benchmarks/warp_pomdp.py --tasks $t        # this repo, on the GPU
  uv run benchmarks/robosuite_baseline.py --tasks $t       # robosuite 1.5.1 on CPU MuJoCo 3.2.6, in its own environment
done
uv run python benchmarks/report.py                         # the tables below, from benchmarks/results/
```

Each run adds its task to `benchmarks/results/<simulator>.json`; one process per task keeps GPU memory from
accumulating across tasks.

`robosuite_baseline.py` declares its own dependencies (PEP 723), so uv runs it in a separate environment with
robosuite 1.5.1 and MuJoCo 3.2.6. Both read the released datasets from `$ROBOMIMIC_DATA`.

## Results

<!-- results -->
Simulators: robosuite 1.5.1, CPU MuJoCo 3.2.6 (48 CPU threads); MuJoCo Warp on NVIDIA GeForce RTX 4090.

### Throughput, env steps per second (step + observe, low-dim observations)

| task | robosuite, 1 process | robosuite, 48 processes | Warp, 1 worlds | Warp, 64 worlds | Warp, 256 worlds | Warp, 1024 worlds | Warp, 4096 worlds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lift (ph) | 199 | 5.8k | 76 | 3.7k | 16.2k | 59.8k | 94.5k |
| Can (ph) | 163 | 4.4k | 56 | 3.8k | 16.3k | 42.6k | 73.6k |
| Square (mh) | 139 | 3.8k | 60 | 5.7k | 20.1k | 53.1k | 76.8k |
| Transport (ph) | 56 | 1.4k | 41 | 3.3k | 10.9k | 21.3k | — |
| Tool Hang (ph) | 174 | 4.2k | 34 | 3.0k | 12.8k | 36.1k | 60.6k |

### Throughput with two 84x84 RGB cameras, env steps per second

| task | robosuite, 1 process | robosuite, 48 processes | Warp, 64 worlds | Warp, 256 worlds | Warp, 1024 worlds |
| --- | --- | --- | --- | --- | --- |
| Lift (ph) | 198 | 6.0k | 3.1k | 12.0k | 18.5k |
| Can (ph) | 143 | 5.1k | 2.6k | 7.1k | 10.7k |
| Square (mh) | 140 | 4.9k | 4.1k | 10.7k | 15.1k |
| Transport (ph) | 51 | 1.7k | 2.0k | 4.7k | 4.5k |
| Tool Hang (ph) | 152 | 4.6k | 2.0k | 6.6k | 9.1k |

### Fidelity to the released demonstrations

Open-loop replay of each demo's recorded actions from its first state, and state errors after 1 and 10 recorded actions from recorded states (median / 95th percentile over 20 states per demo).

| task | simulator | replay success | object error, 1 step (mm) | object error, 10 steps (mm) | arm joint error, 1 step (mrad) | arm joint error, 10 steps (mrad) |
| --- | --- | --- | --- | --- | --- | --- |
| Lift (ph) | robosuite | 16/16 | 0.33 / 1.99 | 0.74 / 1.76 | 0.20 / 1.07 | 7.12 / 16.79 |
| Lift (ph) | Warp | 16/16 | 0.08 / 0.69 | 0.11 / 2.06 | 0.38 / 1.18 | 8.56 / 17.11 |
| Can (ph) | robosuite | 15/16 | 0.01 / 0.05 | 0.08 / 8.03 | 0.16 / 0.78 | 5.76 / 20.08 |
| Can (ph) | Warp | 8/8 | 0.21 / 0.32 | 2.29 / 6.67 | 0.55 / 4.46 | 9.55 / 22.73 |
| Square (mh) | robosuite | 13/16 | 0.00 / 0.04 | 0.01 / 2.25 | 0.13 / 0.80 | 4.93 / 18.62 |
| Square (mh) | Warp | 7/8 | 0.20 / 0.73 | 0.68 / 6.04 | 0.32 / 3.22 | 7.92 / 22.15 |
| Transport (ph) | robosuite | 12/16 | 0.04 / 1.74 | 0.36 / 5.24 | 0.24 / 3.04 | 9.19 / 18.90 |
| Transport (ph) | Warp | 8/16 | 0.18 / 1.53 | 1.74 / 12.94 | 0.44 / 3.96 | 10.40 / 21.13 |
| Tool Hang (ph) | robosuite | 0/16 | 0.00 / 0.02 | 0.03 / 6.78 | 0.11 / 0.65 | 4.08 / 17.99 |
| Tool Hang (ph) | Warp | 0/16 | 0.14 / 0.74 | 0.90 / 20.48 | 0.30 / 1.61 | 6.65 / 18.48 |

### Transferred demonstrations (datasets/warp)

Every released demo re-simulated in the Warp POMDP by `robomimic.data.transfer`; deviation is the largest distance of any object from its recorded position over the demo (median / 95th percentile / max).

| task | demos | successful in Warp | largest object deviation (mm) | final object deviation (mm) |
| --- | --- | --- | --- | --- |
| Lift (ph) | 200 | 200 | 5.2 / 8.9 / 13.9 | 3.7 / 6.7 / 13.3 |
| Can (ph) | 200 | 200 | 48.0 / 83.8 / 148.1 | 11.0 / 25.0 / 67.8 |
| Square (mh) | 300 | 300 | 11.3 / 81.7 / 136.0 | 3.1 / 7.6 / 77.8 |
| Transport (ph) | 200 | 200 | 30.7 / 167.8 / 843.1 | 15.4 / 91.0 / 839.5 |
| Tool Hang (ph) | 200 | 200 | 22.5 / 81.1 / 142.2 | 12.4 / 57.3 / 99.1 |
<!-- /results -->

## Notes

- robosuite ran on a 48-thread CPU machine and Warp on a separate machine's RTX 4090.
- A single Warp world is slower than one robosuite process (GPU launch latency); Warp overtakes one robosuite
  process below 64 worlds and all 48 CPU threads between 64 and 256 worlds.
- Transport is measured up to 1024 worlds: at 4096, MuJoCo Warp fails to allocate its contact buffers for this
  scene on the 24 GB GPU.
- Warp's fidelity for Can and Square uses 8 demos and 10 recorded states per demo; the other rows use 16 and 20.
- Open-loop replay is chaotic in both simulators: robosuite itself completes 0 (Tool Hang) to 16 (Lift) of 16 demos.
  The transferred demos were made because replaying the recorded actions does not reproduce them.
- Transferred demos track the recorded trajectories closely for most demos, but some succeed along a
  different path: an object is more than 10 cm from its recorded position at some point in 20 of 200 Transport,
  12 of 300 Square, 4 of 200 Tool Hang, and 3 of 200 Can demos.
