"""
Replay dataset demonstrations open-loop in the MuJoCo Warp POMDP and save videos of one camera, rendered by
the MuJoCo Warp ray tracer. With --show_dataset, each frame shows the dataset's recorded state alongside,
rendered by MuJoCo's OpenGL renderer, which needs a GL backend (MUJOCO_GL=egl without a display).

Example:
    python -m robomimic.scripts.replay_demos --dataset ~/data/robomimic_v15/lift/ph/demo_v15.hdf5 --demos 0 1 --output_dir videos
"""
import argparse
from pathlib import Path

import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw

import jax
import jax.numpy as jnp

from robomimic.data import hdf5
from robomimic.env import TASK_OF_ENV_NAME, Pixels, RobomimicPOMDP


def _label(image, text):
    image = Image.fromarray(image)
    ImageDraw.Draw(image).text((6, 4), text, fill=(255, 255, 0))
    return np.asarray(image)


def replay(f, demo, task, camera, size, output_dir, show_dataset):
    states = f[f"data/{demo}/states"][()]
    actions = f[f"data/{demo}/actions"][()]
    pomdp = RobomimicPOMDP(
        task, model_xml=f[f"data/{demo}"].attrs["model_file"], observation=Pixels((camera,), size, size), max_worlds=1,
    )
    step = jax.jit(lambda s, a: pomdp.step(None, s, a))
    image = jax.jit(lambda s: pomdp.observe(None, s, None))
    success = jax.jit(pomdp.success)

    state = pomdp.state_from_flat(states[0])
    succeeded = False
    frames = []
    for t in range(len(actions) + 1):
        frame = _label(np.asarray(image(state)), f"Warp replay t={t} success={succeeded}")
        if show_dataset:
            recorded = pomdp.render_flat(states[min(t, len(states) - 1)], camera_name=camera, height=size, width=size)
            frame = np.concatenate([frame, _label(recorded, "dataset")], 1)
        frames.append(frame)
        if t < len(actions):
            state = step(state, jnp.asarray(actions[t], jnp.float32))
            succeeded |= bool(success(state))

    path = output_dir / f"{demo}_{camera}.mp4"
    imageio.mimsave(path, frames, fps=20)
    print(f"{demo}: open-loop success {succeeded} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="path to hdf5 dataset")
    parser.add_argument("--demos", type=int, nargs="+", default=[0], help="indices of the demos to replay")
    parser.add_argument("--camera", default="agentview", help="camera to render")
    parser.add_argument("--size", type=int, default=256, help="image height and width in pixels")
    parser.add_argument("--output_dir", type=Path, required=True, help="directory for the videos")
    parser.add_argument("--show_dataset", action="store_true", help="show the dataset's recorded states alongside")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.dataset.expanduser()
    task = TASK_OF_ENV_NAME[hdf5.get_env_metadata_from_dataset(path)["env_name"]]
    with h5py.File(path, "r") as f:
        for i in args.demos:
            replay(f, f"demo_{i}", task, args.camera, args.size, args.output_dir, args.show_dataset)


if __name__ == "__main__":
    main()
