"""Render the task images of `docs/tasks.md`: the first and last state of each task's first released demo.

    MUJOCO_GL=egl uv run python docs/images.py
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from robomimic.data import registry
from robomimic.env import RobomimicPOMDP

DATASETS = {"lift": "lift/ph", "can": "can/ph", "square": "square/mh", "transport": "transport/ph", "tool_hang": "tool_hang/ph"}
CAMERAS = {"tool_hang": "sideview"}


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    for task, dataset in DATASETS.items():
        with h5py.File(registry.DATA_DIR / dataset / "demo_v15.hdf5", "r") as f:
            demo = f["data/demo_0"]
            states, model = demo["states"][()], demo.attrs["model_file"]
        env = RobomimicPOMDP(task, model_xml=model, max_worlds=1)
        camera = CAMERAS.get(task, "agentview")
        frames = [env.render_flat(states[t], camera_name=camera, height=args.size, width=args.size) for t in (0, -1)]
        Image.fromarray(np.concatenate(frames, axis=1)).save(args.out / f"{task}.png", optimize=True)
        print(f"wrote {args.out / f'{task}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "images")
    parser.add_argument("--size", type=int, default=256)
    main(parser.parse_args())
