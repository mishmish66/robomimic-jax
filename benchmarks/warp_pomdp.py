"""
This repo's simulation: the robomimic tasks as jax_pomdps POMDPs on MuJoCo Warp. Measures simulation throughput
at growing numbers of parallel worlds (with and without cameras), fidelity to the released demonstrations in
the same way as `robosuite_baseline.py` (open-loop replay success, and one-step and 10-step state errors from
recorded states), and how closely the transferred demonstrations in datasets/warp track the recorded ones,
writing them to a JSON file for `report.py`.

    uv run python benchmarks/warp_pomdp.py --output benchmarks/results/warp.json
"""
import argparse
import gc
import json
import time
from pathlib import Path

import h5py
import numpy as np

import jax
import jax.numpy as jnp
import mujoco

from robomimic.data import registry
from robomimic.env import RobomimicPOMDP, load_model, parked_joints

DATASETS = {"lift": "lift/ph", "can": "can/ph", "square": "square/mh", "transport": "transport/ph", "tool_hang": "tool_hang/ph"}
CAMERAS = {"transport": ["shouldercamera0", "robot0_eye_in_hand"], "tool_hang": ["sideview", "robot0_eye_in_hand"]}
PACKED = Path(__file__).parent.parent / "datasets" / "warp"


def dataset_path(task):
    return registry.DATA_DIR / DATASETS[task] / "demo_v15.hdf5"


def throughput(task, worlds, steps, cameras):
    """Env steps (step and observe) per second of `worlds` worlds stepped together."""
    # free earlier environments' GPU resources now, not during this one's graph capture
    gc.collect()
    names = CAMERAS.get(task, ["agentview", "robot0_eye_in_hand"]) if cameras else ()
    env = RobomimicPOMDP(task, camera_names=names, max_worlds=worlds)
    keys = jax.random.split(jax.random.key(0), worlds)
    state = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.vmap(env.action_space.sample)(keys) * 0.3

    @jax.jit
    def step(state):
        state = jax.vmap(env.step)(keys, state, actions)
        return state, jax.vmap(env.observe)(keys, state, actions)

    jax.block_until_ready(step(state))
    start = time.perf_counter()
    for _ in range(steps):
        state, obs = step(state)
    jax.block_until_ready(obs)
    return worlds * steps / (time.perf_counter() - start)


def _object_qpos(m, flat):
    """qpos indices of the positions of the objects that are not parked at flattened state `flat`."""
    live = np.setdiff1d(np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE), parked_joints(m, flat[1:1 + m.nq]))
    return np.concatenate([m.jnt_qposadr[j] + np.arange(3) for j in live])


def _grip_states(env, actions):
    """Gripper integrator state before each action."""
    grip = jnp.zeros((len(env.arms), 2), jnp.float32)
    grips = [grip]
    for a in jnp.asarray(actions, jnp.float32):
        grip = jnp.stack([arm.step_grip(g, a) for arm, g in zip(env.arms, grip)])
        grips.append(grip)
    return jnp.stack(grips)


def fidelity(task, demos, windows):
    """Open-loop replay success, and state errors after 1 and 10 steps from recorded states, over `demos` demos."""
    successes, errors = [], {1: dict(arm=[], obj=[]), 10: dict(arm=[], obj=[])}
    with h5py.File(dataset_path(task), "r") as f:
        names = sorted(f["data"], key=lambda k: int(k[5:]))[:demos]
        for name in names:
            demo = f[f"data/{name}"]
            states, actions = demo["states"][()], demo["actions"][()]
            env = RobomimicPOMDP(task, model_xml=demo.attrs["model_file"], max_worlds=windows)
            m = env.model
            arm_qpos = np.concatenate([arm.qpos for arm in env.arms])
            obj_qpos = _object_qpos(m, states[0])

            step = jax.jit(lambda s, a: (s := env.step(None, s, a), env.success(s)))
            state, succeeded = env.state_from_flat(states[0]), False
            for a in actions:
                state, success = step(state, jnp.asarray(a, jnp.float32))
                succeeded |= bool(success)
            successes.append(succeeded)

            grips = _grip_states(env, actions)
            q0 = tuple(jnp.broadcast_to(jnp.asarray(states[0][1:][arm.qpos], jnp.float32), (windows, len(arm.qpos))) for arm in env.arms)
            starts = np.linspace(0, len(actions) - 11, windows).astype(int)
            batch = jax.jit(jax.vmap(env.step))
            for horizon in (1, 10):
                s = jax.jit(jax.vmap(env.state_from_flat))(jnp.asarray(states[starts], jnp.float32))._replace(grip=grips[starts], q0=q0)
                for k in range(horizon):
                    s = batch(None, s, jnp.asarray(actions[starts + k], jnp.float32))
                q, target = np.asarray(s.data.qpos), states[starts + horizon][:, 1:1 + m.nq]
                errors[horizon]["arm"] += list(np.abs(q[:, arm_qpos] - target[:, arm_qpos]).max(1))
                errors[horizon]["obj"] += list(np.linalg.norm((q[:, obj_qpos] - target[:, obj_qpos]).reshape(windows, -1, 3), axis=2).max(1))
            print(f"[warp {task}] {name}: replay success {succeeded}", flush=True)
    summary = {f"{h}_step": {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in e.items()}
               for h, e in errors.items()}
    return dict(replay_success=int(sum(successes)), replay_demos=len(successes), **summary)


def transferred(task):
    """Success of the transferred demos, and how far their objects get from the recorded trajectories."""
    name = DATASETS[task].replace("/", "_")
    with np.load(PACKED / f"{name}.npz") as packed, h5py.File(dataset_path(task), "r") as f:
        starts = np.concatenate([[0], np.cumsum(packed["lengths"])])
        models = {}
        largest, final = [], []
        for i, demo in enumerate(packed["names"]):
            xml = str(packed["models"][packed["model_index"][i]])
            m = models.setdefault(xml, load_model(task, xml))
            recorded = f[f"data/{demo}/states"][()]
            simulated = packed["step/states"][starts[i]:starts[i + 1]]
            obj = 1 + _object_qpos(m, recorded[0])
            distance = np.linalg.norm((simulated[:, obj] - recorded[:, obj]).reshape(len(recorded), -1, 3), axis=2).max(1)
            largest.append(float(distance.max()))
            final.append(float(distance[-1]))
        summary = lambda x: dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
        return dict(demos=int(len(packed["names"])), successful=int(packed["success"].sum()),
                    largest_object_deviation=summary(largest), final_object_deviation=summary(final))

def main(args):
    output = Path(args.output)
    results = json.loads(output.read_text()) if output.exists() else {}
    results["simulator"] = f"MuJoCo Warp on {jax.devices()[0].device_kind}"
    tasks = results.setdefault("tasks", {})
    for task in args.tasks:
        start = time.time()
        tasks[task] = dict(
            low_dim={str(n): throughput(task, n, args.steps, False) for n in args.worlds},
            cameras={str(n): throughput(task, n, args.steps, True) for n in args.camera_worlds},
            **fidelity(task, args.demos, args.windows),
            transferred=transferred(task),
        )
        print(f"[warp {task}] {json.dumps(tasks[task])} ({time.time() - start:.0f}s)", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="results file; tasks are added to it", default="benchmarks/results/warp.json")
    parser.add_argument("--tasks", nargs="+", default=list(DATASETS))
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 64, 256, 1024, 4096])
    parser.add_argument("--camera_worlds", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--steps", type=int, default=20, help="steps timed per measurement")
    parser.add_argument("--demos", type=int, default=16, help="demos replayed per task")
    parser.add_argument("--windows", type=int, default=20, help="recorded states per demo the state errors start from")
    main(parser.parse_args())
