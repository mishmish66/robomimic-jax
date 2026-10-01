"""
Add the POMDP's observations of each recorded state to a dataset: "obs" (before each action) and "next_obs"
(after it), as in the robomimic low_dim datasets. The state after the last action is not recorded, so the
last "next_obs" observes the simulated outcome of the last action.

Example:
    python add_observations.py --dataset ~/data/robomimic_warp/lift/ph/demo_v15.hdf5
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import jax
import jax.numpy as jnp

from robomimic.pomdp import TASK_OF_ENV_NAME, RobomimicPOMDP


class Observer:
    """Jitted observations of batches of flattened states, each with the state recorded before it."""
    def __init__(self, env, batch):
        self.env, self.batch = env, batch

        def observe(previous, flat):
            cache = env.state_from_flat(previous).cache
            return env.observe(None, env.state_from_flat(flat)._replace(cache=cache), None)

        def outcome(states, actions):
            """Flattened state after the last of @actions, taken from the last of @states."""
            grip = jnp.zeros((len(env._arms), 2), jnp.float32)
            for action in actions[:-1]:
                grip = jnp.stack([arm.step_grip(g, action) for arm, g in zip(env._arms, grip)])
            q0 = tuple(states[0][1:][arm.qpos] for arm in env._arms)
            state = env.state_from_flat(states[-1])._replace(grip=grip, q0=q0)
            d = env.step(None, state, actions[-1]).data
            return jnp.concatenate([d.time[None], d.qpos, d.qvel])

        self._observe = jax.jit(jax.vmap(observe))
        self.outcome = jax.jit(outcome)

    def __call__(self, previous, flats):
        """Observations of @flats, given the states @previous recorded before them, as numpy arrays."""
        parts = []
        for i in range(0, len(flats), self.batch):
            p, f = previous[i:i + self.batch], flats[i:i + self.batch]
            n = len(f)
            p, f = (np.concatenate([x, np.repeat(x[-1:], self.batch - n, 0)]).astype(np.float32) for x in (p, f))
            parts.append({k: np.asarray(v)[:n] for k, v in self._observe(jnp.asarray(p), jnp.asarray(f)).items()})
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main(args):
    with h5py.File(Path(args.dataset).expanduser(), "a") as f:
        task = TASK_OF_ENV_NAME[json.loads(f["data"].attrs["env_args"])["env_name"]]
        names = sorted(f["data"], key=lambda k: int(k.removeprefix("demo_")))
        observers = {}
        for i, name in enumerate(names):
            demo = f[f"data/{name}"]
            model = demo.attrs["model_file"]
            if model not in observers:
                observers.clear()
                observers[model] = Observer(RobomimicPOMDP(task, model_xml=model, max_worlds=args.batch), args.batch)
            states = demo["states"][()]
            last = observers[model].outcome(jnp.asarray(states, jnp.float32), jnp.asarray(demo["actions"][()], jnp.float32))
            nexts = np.concatenate([states[1:], np.asarray(last, states.dtype)[None]])
            previous = np.concatenate([states[:1], states[:-1]])
            for group, (before, flats) in {"obs": (previous, states), "next_obs": (states, nexts)}.items():
                if group in demo:
                    del demo[group]
                for k, v in observers[model](before, flats).items():
                    demo.create_dataset(f"{group}/{k}", data=v)
            if (i + 1) % 20 == 0 or i + 1 == len(names):
                print(f"[{task}] observations of {i + 1}/{len(names)} demos", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="hdf5 dataset with recorded states")
    parser.add_argument("--batch", type=int, default=256, help="states observed together")
    main(parser.parse_args())
