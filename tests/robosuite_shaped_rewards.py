# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["robosuite==1.5.1", "mujoco==3.2.6", "h5py", "numpy<2.3"]
# ///
"""
Record robosuite's shaped rewards (`reward_shaping=True`) at every recorded state of the first demos of each
released dataset, in the demo's own model, to robosuite_shaped_rewards.json for `test_env.py`.

    uv run tests/robosuite_shaped_rewards.py
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import robosuite

DATA_DIR = Path(os.environ.get("ROBOMIMIC_DATA", "~/data/robomimic_v15")).expanduser()
DATASETS = {"lift": "lift/ph", "can": "can/ph", "square": "square/mh", "transport": "transport/ph", "tool_hang": "tool_hang/ph"}


def main(args):
    rewards = {}
    for task, dataset in DATASETS.items():
        with h5py.File(DATA_DIR / dataset / "demo_v15.hdf5", "r") as f:
            meta = json.loads(f["data"].attrs["env_args"])
            demos = {n: (f[f"data/{n}"].attrs["model_file"], f[f"data/{n}/states"][()]) for n in args.demos}
        kwargs = dict(meta["env_kwargs"], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                      ignore_done=True, reward_shaping=True)
        env = robosuite.make(meta["env_name"], **kwargs)
        env.reset()
        # tasks without shaped rewards switch to sparse rewards on their first reward
        env.reward()
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
