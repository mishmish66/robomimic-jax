# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["robosuite==1.5.1", "mujoco==3.2.6", "h5py", "numpy<2.3"]
# ///
"""
Original robomimic simulation: robosuite v1.5.1 on CPU MuJoCo, as robomimic builds it from a dataset's metadata.
Measures simulation throughput (one process, and one process per CPU core, with and without cameras) and
fidelity to the released demonstrations (open-loop replay success, and one-step and 10-step state errors from
recorded states), writing them to a JSON file for `report.py`.

    uv run benchmarks/robosuite_baseline.py --output benchmarks/results/robosuite.json
"""
import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

DATA_DIR = Path(os.environ.get("ROBOMIMIC_DATA", "~/data/robomimic_v15")).expanduser()
DATASETS = {"lift": "lift/ph", "can": "can/ph", "square": "square/mh", "transport": "transport/ph", "tool_hang": "tool_hang/ph"}
CAMERAS = {"transport": ["shouldercamera0", "robot0_eye_in_hand"], "tool_hang": ["sideview", "robot0_eye_in_hand"]}
# parked objects are those farther than this from the world origin
PARKED_DISTANCE = 5.


def dataset_path(task):
    return DATA_DIR / DATASETS[task] / "demo_v15.hdf5"


def make_env(task, cameras=False):
    import robosuite
    with h5py.File(dataset_path(task), "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=cameras, ignore_done=True, use_object_obs=True,
                  use_camera_obs=cameras, camera_depths=False)
    if cameras:
        kwargs.update(camera_names=CAMERAS.get(task, ["agentview", "robot0_eye_in_hand"]), camera_heights=84,
                      camera_widths=84)
    return robosuite.make(env_meta["env_name"], **kwargs)


def _steps_per_second(task, cameras, steps, seed=0):
    env = make_env(task, cameras)
    env.reset()
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1., 1., (steps, env.action_dim)) * 0.3
    env.step(actions[0])
    start = time.perf_counter()
    for a in actions:
        env.step(a)
    return steps / (time.perf_counter() - start)


def _worker(job):
    return _steps_per_second(*job)


def throughput(task, cameras, steps, workers):
    """Env steps per second of one process, and summed over `workers` processes."""
    single = _steps_per_second(task, cameras, steps)
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        parallel = sum(pool.map(_worker, [(task, cameras, steps, seed) for seed in range(workers)]))
    return dict(single=single, parallel=parallel, workers=workers)


def _start_episode(env, flat, grip=None):
    """Sets `env` to flattened state `flat`, with its controllers as in a demo that started at `flat`."""
    env.sim.set_state_from_flattened(flat)
    env.sim.forward()
    arms = [(robot, arm) for robot in env.robots for arm in robot.arms]
    for robot in env.robots:
        for controller in robot.part_controllers.values():
            controller.new_update = True
    for i, (robot, arm) in enumerate(arms):
        robot.gripper[arm].current_action = np.zeros(2) if grip is None else grip[i]


def _object_qpos(m, flat):
    free = [j for j in range(m.njnt) if m.jnt_type[j] == 0]
    live = [j for j in free if np.linalg.norm(flat[1 + m.jnt_qposadr[j]:4 + m.jnt_qposadr[j]]) <= PARKED_DISTANCE]
    return np.array([m.jnt_qposadr[j] + k for j in live for k in range(3)])


def _grip_states(actions, arms, speed):
    """robosuite's Panda gripper integrator state before each action."""
    grips = [np.zeros((arms, 2))]
    for a in actions:
        g = grips[-1].copy()
        for i in range(arms):
            g[i] = np.clip(g[i] + np.array([-1., 1.]) * speed * np.sign(a[7 * i + 6]), -1., 1.)
        grips.append(g)
    return grips


def fidelity(task, demos, windows):
    """Open-loop replay success, and state errors after 1 and 10 steps from recorded states, over `demos` demos."""
    env = make_env(task)
    env.reset()
    successes, errors = [], {1: dict(arm=[], obj=[]), 10: dict(arm=[], obj=[])}
    with h5py.File(dataset_path(task), "r") as f:
        names = sorted(f["data"], key=lambda k: int(k[5:]))[:demos]
        for name in names:
            demo = f[f"data/{name}"]
            states, actions = demo["states"][()], demo["actions"][()]
            env.reset_from_xml_string(env.edit_model_xml(demo.attrs["model_file"]))
            env.reset()
            m = env.sim.model._model
            arm_qpos = np.concatenate([1 + np.asarray(r._ref_joint_pos_indexes) for r in env.robots])
            obj_qpos = 1 + _object_qpos(m, states[0])
            speed = env.robots[0].gripper[env.robots[0].arms[0]].speed
            grips = _grip_states(actions, len(env.robots), speed)

            _start_episode(env, states[0])
            for robot in env.robots:
                for part, controller in robot.part_controllers.items():
                    if part in robot.arms:
                        controller.initial_joint = np.array(states[0][1:][controller.qpos_index])
            succeeded = False
            for a in actions:
                env.step(a)
                succeeded |= bool(env._check_success())
            successes.append(succeeded)

            starts = np.linspace(0, len(actions) - 11, windows).astype(int)
            for horizon in (1, 10):
                for t in starts:
                    _start_episode(env, states[t], grips[t])
                    for a in actions[t:t + horizon]:
                        env.step(a)
                    q, target = env.sim.get_state().flatten(), states[t + horizon]
                    errors[horizon]["arm"].append(float(np.abs(q[arm_qpos] - target[arm_qpos]).max()))
                    errors[horizon]["obj"].append(float(np.linalg.norm((q[obj_qpos] - target[obj_qpos]).reshape(-1, 3), axis=1).max()))
            print(f"[robosuite {task}] {name}: replay success {succeeded}", flush=True)
    summary = {f"{h}_step": {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in e.items()}
               for h, e in errors.items()}
    return dict(replay_success=int(sum(successes)), replay_demos=len(successes), **summary)


def main(args):
    output = Path(args.output)
    results = json.loads(output.read_text()) if output.exists() else {}
    results.update(simulator="robosuite 1.5.1, CPU MuJoCo 3.2.6", cpu=os.cpu_count())
    tasks = results.setdefault("tasks", {})
    for task in args.tasks:
        start = time.time()
        tasks[task] = dict(
            low_dim=throughput(task, False, args.steps, args.workers),
            cameras=throughput(task, True, args.steps // 4, args.workers),
            **fidelity(task, args.demos, args.windows),
        )
        print(f"[robosuite {task}] {json.dumps(tasks[task])} ({time.time() - start:.0f}s)", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="results file; tasks are added to it", default="benchmarks/results/robosuite.json")
    parser.add_argument("--tasks", nargs="+", default=list(DATASETS))
    parser.add_argument("--steps", type=int, default=400, help="steps timed per process")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes of the parallel measurement")
    parser.add_argument("--demos", type=int, default=16, help="demos replayed per task")
    parser.add_argument("--windows", type=int, default=20, help="recorded states per demo the state errors start from")
    main(parser.parse_args())
