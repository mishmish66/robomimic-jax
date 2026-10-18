# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "robosuite @ git+https://github.com/ARISE-Initiative/robosuite@b9d8d3de5e3dfd1724f4a0e6555246c460407daa",
#     "mimicgen @ git+https://github.com/NVlabs/mimicgen@72bd767c255545f462e7ccfb2731f2e5d4c1d9bb",
#     "mujoco==2.3.2", "h5py", "numpy<2", "numba", "scipy", "termcolor", "pillow", "huggingface_hub",
# ]
# ///
"""
Record robosuite's shaped rewards (`reward_shaping=True`) at every recorded state of the first demos of each ported
MimicGen dataset, in the demo's own model, into robosuite_shaped_rewards.json for `test_env.py`. MimicGen's tasks run
on robosuite v1.4 at the commit MimicGen pins.

    uv run tests/mimicgen_shaped_rewards.py
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import robosuite
import mimicgen.envs.robosuite  # noqa: F401  registers MimicGen's environments

DATA_DIR = Path(os.environ.get("MIMICGEN_DATA", "~/data/mimicgen")).expanduser()
DATASETS = {task: f"core/{task}.hdf5" for task in ("stack_d0", "coffee_d0", "threading_d0")}


def main(args):
    rewards = json.loads(args.output.read_text()) if args.output.exists() else {}
    for task, dataset in DATASETS.items():
        with h5py.File(DATA_DIR / dataset, "r") as f:
            meta = json.loads(f["data"].attrs["env_args"])
            demos = {n: (f[f"data/{n}"].attrs["model_file"], f[f"data/{n}/states"][()]) for n in args.demos}
        kwargs = dict(meta["env_kwargs"], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                      ignore_done=True, reward_shaping=True)
        env = robosuite.make(meta["env_name"], **kwargs)
        env.reset()
        rewards[task] = {}
        for name, (xml, states) in demos.items():
            env.reset_from_xml_string(env.edit_model_xml(xml))
            demo_rewards = []
            for flat in states:
                env.sim.set_state_from_flattened(flat)
                env.sim.forward()
                demo_rewards.append(round(float(env.reward()), 7))
            rewards[task][name] = demo_rewards
        print(f"{task}: {', '.join(f'{n} ({len(r)} states)' for n, r in rewards[task].items())}", flush=True)
    args.output.write_text(json.dumps(rewards) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", nargs="+", default=["demo_0", "demo_1"])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "robosuite_shaped_rewards.json")
    main(parser.parse_args())
