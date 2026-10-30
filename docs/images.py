"""Render the task images of `docs/tasks.md`: the first and last state of each task's first released demo, as the
pixels mode observes them.

    uv run --extra video python docs/images.py
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import jax

from robomimic.data import registry
from robomimic.env import Pixels, RobomimicPOMDP

DATASETS = {"lift": "lift/ph", "can": "can/ph", "square": "square/mh", "transport": "transport/ph", "tool_hang": "tool_hang/ph"}
# one variant of each MimicGen task family, named after the family
MIMICGEN = {
    "stack": "stack_d0", "stack_three": "stack_three_d0", "square_d": "square_d0", "threading": "threading_d0",
    "coffee": "coffee_d0", "three_piece_assembly": "three_piece_assembly_d0", "hammer_cleanup": "hammer_cleanup_d0",
    "mug_cleanup": "mug_cleanup_d0", "kitchen": "kitchen_d0", "nut_assembly": "nut_assembly_d0",
    "pick_place": "pick_place_d0", "coffee_preparation": "coffee_preparation_d0",
}
CAMERAS = {"tool_hang": "sideview"}


def _demo(path):
    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        return demo["states"][()], demo.attrs["model_file"]


def _render(task, path, out, size):
    states, model = _demo(path)
    camera = CAMERAS.get(task, "agentview")
    env = RobomimicPOMDP(task, model_xml=model, observation=Pixels((camera,), size, size), max_worlds=1)
    observe = jax.jit(lambda flat: env.observe(None, env.state_from_flat(flat), None))
    frames = [np.asarray(observe(states[t])) for t in (0, -1)]
    Image.fromarray(np.concatenate(frames, axis=1)).save(out, optimize=True)
    print(f"wrote {out}")


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    for task, dataset in DATASETS.items():
        _render(task, registry.DATA_DIR / dataset / "demo_v15.hdf5", args.out / f"{task}.png", args.size)
    for name, task in MIMICGEN.items():
        _render(task, registry.MIMICGEN_DATA_DIR / registry.MIMICGEN_DATASETS[task], args.out / f"{name}.png", args.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "images")
    parser.add_argument("--size", type=int, default=256)
    main(parser.parse_args())
