"""
MimicGen's data generation in the MuJoCo Warp POMDP.

A few source demonstrations, transferred into the POMDP, are split into object-centric subtask segments. Each
generated demo starts from a reset of the target task variant, takes one source demo, and replays its segments with
the gripper's target poses carried into the frame of each segment's object as it lies when the segment starts,
joined by short interpolations, through the arm controller with action noise. Demos that succeed are kept. Many
demos are generated together, one per world.
"""
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import h5py
import mujoco
import numpy as np

import jax
import jax.numpy as jnp

from robomimic.env import TASK_OF_ENV_NAME, RobomimicPOMDP, load_model, task_spec
from robomimic.env.geometry import axisangle2mat, mat2quat
from robomimic.env.pomdp import _ASSETS


class Subtask(NamedTuple):
    """One object-centric segment of MimicGen's task spec."""
    body: str | None  # body whose frame the segment's gripper poses follow, or None for the world frame
    signal: str | None  # signal whose first 0 -> 1 transition ends the segment in a source demo; None for the last
    offsets: tuple[int, int] = (0, 0)  # range of the random shift of the segment's end, in steps
    noise: float = 0.05  # standard deviation of the noise added to the arm actions of the segment
    interpolation: int = 5  # steps interpolating from the previous segment's last target pose to this one's first
    nearest: int = 0  # pick the source demo among this many with the closest object poses; 0 picks at random


def _grasp(m, d, groups, geoms):
    """Whether every geom group of `groups` touches a geom of `geoms`, as robosuite's `_check_grasp` tests."""
    ids = {m.geom(g).id for g in geoms}
    touching = set()
    for c in d.contact[:d.ncon]:
        if c.geom1 in ids:
            touching.add(c.geom2)
        elif c.geom2 in ids:
            touching.add(c.geom1)
    return all(any(m.geom(g).id in touching for g in group) for group in groups)


def _tolerant_grasp(m, d, geoms):
    """MimicGen's `_check_grasp_tolerant`: both pads, or a pad and the opposite finger, touch `geoms`."""
    return any(_grasp(m, d, groups, geoms) for groups in (
        [["gripper0_finger1_pad_collision"], ["gripper0_finger2_pad_collision"]],
        [["gripper0_finger1_collision"], ["gripper0_finger2_pad_collision"]],
        [["gripper0_finger2_collision"], ["gripper0_finger1_pad_collision"]],
    ))


def _contact(m, d, geoms_a, geoms_b):
    """Whether any geom of `geoms_a` touches any geom of `geoms_b`, as robosuite's `check_contact` tests."""
    a, b = {m.geom(g).id for g in geoms_a}, {m.geom(g).id for g in geoms_b}
    return any((c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a) for c in d.contact[:d.ncon])


def _body_geoms(m, body):
    """The colliding geoms of the bodies under `body`."""
    root = m.body(body).id
    under = lambda b: b == root or (b != 0 and under(m.body_parentid[b]))
    return [m.geom(g).name for g in range(m.ngeom) if under(m.geom_bodyid[g]) and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _qpos(m, d, joint):
    return d.qpos[m.jnt_qposadr[m.joint(joint).id]]


def _xpos(m, d, body):
    return d.xpos[m.body(body).id]


def _drawer_signals(link):
    """The drawer opened past 10 cm with the gripper 24 cm away from it, as MimicGen's MugCleanup and HammerCleanup."""
    def signal(m, d, spec, eef_pos):
        return _qpos(m, d, spec["slide"]) < -0.10 and np.linalg.norm(eef_pos - _xpos(m, d, link)) > 0.24
    return signal


def _mug_cleanup_signals(m, d, spec, eef_pos):
    return {
        "open": _drawer_signals("DrawerObject_drawer_link")(m, d, spec, eef_pos),
        "grasp": _tolerant_grasp(m, d, spec["mug_geoms"]),
    }


def _hammer_cleanup_signals(m, d, spec, eef_pos):
    return {
        "open": _drawer_signals("CabinetObject_drawer_link")(m, d, spec, eef_pos),
        "grasp": _grasp(m, d, spec["pad_geoms"], spec["hammer_geoms"]),
    }


def _coffee_signals(m, d, spec, eef_pos):
    return {"grasp": _grasp(m, d, spec["pad_geoms"], spec["pod_geoms"])}


def _coffee_preparation_signals(m, d, spec, eef_pos):
    mug_upright = 1. - d.xmat[m.body(spec["mug"]).id].reshape(3, 3)[2, 2] < 1e-3
    mug_on_machine = _contact(m, d, [spec["base_geom"]], spec["mug_geoms"])
    return {
        "mug_grasp": _tolerant_grasp(m, d, spec["mug_geoms"]),
        "mug_place": bool(mug_upright and mug_on_machine and _qpos(m, d, spec["hinge"]) > 2.08),
        "drawer_open": _qpos(m, d, spec["drawer_slide"]) < -0.19,
        "pod_grasp": _grasp(m, d, spec["pad_geoms"], spec["pod_geoms"]),
    }


def _threading_signals(m, d, spec, eef_pos):
    return {"grasp": _grasp(m, d, spec["pad_geoms"], spec["needle_geoms"])}


def _stack_signals(m, d, spec, eef_pos):
    return {"grasp": _grasp(m, d, spec["pad_geoms"], spec["cube_a_geoms"])}


def _stack_three_signals(m, d, spec, eef_pos):
    grasping_a = _grasp(m, d, spec["pad_geoms"], spec["cube_a_geoms"])
    lifted_a = _xpos(m, d, spec["cube_a"])[2] > spec["table_height"] + 0.04
    return {
        "grasp_1": grasping_a,
        "stack_1": bool(not grasping_a and lifted_a and _contact(m, d, spec["cube_a_geoms"], spec["cube_b_geoms"])),
        "grasp_2": _grasp(m, d, spec["pad_geoms"], spec["cube_c_geoms"]),
    }


def _square_signals(m, d, spec, eef_pos):
    return {"grasp": _grasp(m, d, spec["pad_geoms"], spec["object_geoms"][0])}


def _nut_assembly_signals(m, d, spec, eef_pos):
    nut, peg = _xpos(m, d, spec["bodies"][0]), _xpos(m, d, spec["pegs"][0])
    on_peg = abs(nut[0] - peg[0]) < 0.03 and abs(nut[1] - peg[1]) < 0.03 and nut[2] < spec["table_height"] + 0.05
    return {
        "grasp_square_nut": _grasp(m, d, spec["pad_geoms"], spec["object_geoms"][0]),
        "insert_square_nut": bool(on_peg),
        "grasp_round_nut": _grasp(m, d, spec["pad_geoms"], spec["object_geoms"][1]),
    }


_PICK_PLACE_OBJECTS = ("milk", "bread", "cereal", "can")  # in robosuite's object order, as the spec's bodies


def _pick_place_signals(m, d, spec, eef_pos):
    signals = {}
    for i, name in enumerate(_PICK_PLACE_OBJECTS):
        pos = _xpos(m, d, spec["bodies"][i])
        (x_low, y_low), (x_high, y_high) = spec["bin_low"][i], spec["bin_high"][i]
        in_bin = x_low < pos[0] < x_high and y_low < pos[1] < y_high and spec["bin_z"] < pos[2] < spec["bin_z"] + 0.1
        # robosuite counts an object placed once the gripper is away from it
        signals[f"grasp_{name}"] = _grasp(m, d, spec["pad_geoms"], spec["object_geoms"][i])
        signals[f"place_{name}"] = bool(in_bin and np.tanh(10. * np.linalg.norm(eef_pos - pos)) > 0.4)
    return signals


def _three_piece_assembly_signals(m, d, spec, eef_pos):
    grasping_1 = _grasp(m, d, spec["pad_geoms"], spec["piece_1_geoms"])
    offset = _xpos(m, d, spec["piece_1"])[:2] - _xpos(m, d, spec["base"])[:2]
    return {
        "grasp_1": grasping_1,
        "insert_1": bool(np.linalg.norm(offset) < 0.02 and not grasping_1),
        "grasp_2": _grasp(m, d, spec["pad_geoms"], spec["piece_2_geoms"]),
    }


def _kitchen_signals(m, d, spec, eef_pos):
    pot, bread = _body_geoms(m, spec["pot"]), _body_geoms(m, spec["bread"])
    grasped_pot = _grasp(m, d, spec["pad_geoms"], pot)
    offset = np.abs(_xpos(m, d, spec["serving_region"]) - _xpos(m, d, spec["pot"]))
    return {
        "stove_on": _qpos(m, d, spec["button_hinge"]) >= spec["button_on"],
        "grasp_pot": grasped_pot,
        "place_pot_on_stove": bool(_contact(m, d, [spec["pot_bottom"]], [spec["burner"]]) and not grasped_pot),
        "grasp_bread": _grasp(m, d, spec["pad_geoms"], bread),
        "place_bread_in_pot": bool(_contact(m, d, bread, pot) and grasped_pot),
        "serve": bool(offset[0] < 0.05 and offset[1] < 0.10 and offset[2] < 0.05),
    }


# MimicGen's task specs (`mimicgen/exps/templates/robosuite`) and subtask signals (`mimicgen/env_interfaces/robosuite.py`)
SUBTASKS = {
    "coffee": (
        Subtask("coffee_pod_main", "grasp", (5, 10)),
        Subtask("coffee_machine_root", None),
    ),
    "coffee_preparation": (
        Subtask("mug_main", "mug_grasp", (5, 10)),
        Subtask("coffee_machine_root", "mug_place", (5, 10)),
        Subtask("CabinetObject_main", "drawer_open", (5, 10)),
        Subtask("coffee_pod_main", "pod_grasp", (5, 10)),
        Subtask("coffee_machine_root", None),
    ),
    "hammer_cleanup": (
        Subtask("CabinetObject_main", "open", (10, 20)),
        Subtask("hammer_root", "grasp", (10, 20)),
        Subtask("CabinetObject_main", None),
    ),
    "kitchen": (
        Subtask("Button1_main", "stove_on", (10, 20)),
        Subtask("PotObject_root", "grasp_pot", (10, 20)),
        Subtask("Stove1_main", "place_pot_on_stove", (10, 20)),
        Subtask("cube_bread_main", "grasp_bread", (10, 20)),
        Subtask("PotObject_root", "place_bread_in_pot", (10, 20)),
        Subtask("ServingRegionRed_main", "serve", (10, 20)),
        Subtask("Button1_main", None),
    ),
    "mug_cleanup": (
        Subtask("DrawerObject_main", "open", (10, 20)),
        Subtask("cleanup_object_main", "grasp", (10, 20)),
        Subtask("DrawerObject_main", None),
    ),
    "nut_assembly": (
        Subtask("SquareNut_main", "grasp_square_nut", (10, 20), nearest=3),
        Subtask("peg1", "insert_square_nut", (10, 20)),
        Subtask("RoundNut_main", "grasp_round_nut", (10, 20)),
        Subtask("peg2", None),
    ),
    "pick_place": (
        Subtask("Milk_main", "grasp_milk", (10, 20), nearest=3),
        Subtask(None, "place_milk"),
        Subtask("Cereal_main", "grasp_cereal", (10, 20), nearest=3),
        Subtask(None, "place_cereal"),
        Subtask("Bread_main", "grasp_bread", (10, 20), nearest=3),
        Subtask(None, "place_bread"),
        Subtask("Can_main", "grasp_can", (10, 20), nearest=3),
        Subtask(None, None),
    ),
    "square": (
        Subtask("SquareNut_main", "grasp", (10, 20), nearest=3),
        Subtask("peg1", None),
    ),
    "stack": (
        Subtask("cubeA_main", "grasp", (10, 20), nearest=3),
        Subtask("cubeB_main", None, nearest=3),
    ),
    "stack_three": (
        Subtask("cubeA_main", "grasp_1", (10, 20), nearest=3),
        Subtask("cubeB_main", "stack_1", (10, 20), nearest=3),
        Subtask("cubeC_main", "grasp_2", (10, 20), nearest=3),
        Subtask("cubeA_main", None, nearest=3),
    ),
    "threading": (
        Subtask("needle_obj_root", "grasp", (5, 10)),
        Subtask("tripod_obj_root", None),
    ),
    "three_piece_assembly": (
        Subtask("piece_1_root", "grasp_1", (5, 10)),
        Subtask("base_root", "insert_1", (5, 10)),
        Subtask("piece_2_root", "grasp_2", (5, 10)),
        Subtask("piece_1_root", None),
    ),
}
SIGNALS = {
    "coffee": _coffee_signals,
    "coffee_preparation": _coffee_preparation_signals,
    "hammer_cleanup": _hammer_cleanup_signals,
    "kitchen": _kitchen_signals,
    "mug_cleanup": _mug_cleanup_signals,
    "nut_assembly": _nut_assembly_signals,
    "pick_place": _pick_place_signals,
    "square": _square_signals,
    "stack": _stack_signals,
    "stack_three": _stack_three_signals,
    "threading": _threading_signals,
    "three_piece_assembly": _three_piece_assembly_signals,
}


def _family(task):
    return task.rsplit("_d", 1)[0]


def _pose(pos, rot):
    pose = np.zeros(pos.shape[:-1] + (4, 4))
    pose[..., :3, :3], pose[..., :3, 3], pose[..., 3, 3] = rot, pos, 1.
    return pose


def _rotvec_to_mat(v):
    return np.asarray(axisangle2mat(jnp.asarray(v)))


class Source(NamedTuple):
    """A source demo, annotated as MimicGen's `DatagenInfo`: poses are 4 x 4 world-frame transforms per step."""
    eef: np.ndarray  # the grip site
    targets: np.ndarray  # the controller's goal for each action
    grips: np.ndarray  # gripper actions
    objects: dict  # body name -> its poses
    ends: list  # step after which each subtask but the last ends


def annotate(task, path, demos=None):
    """The `Source` of each successful demo of transferred hdf5 dataset `path`, a demo of `task`'s family."""
    family = _family(task)
    subtasks, signals = SUBTASKS[family], SIGNALS[family]
    spec = task_spec(task)
    arm = spec["arms"][0]
    scale_pos, scale_rot = arm["arm_scale"]["output_max"][0], arm["arm_scale"]["output_max"][3]
    bodies = sorted({s.body for s in subtasks if s.body is not None})
    sources = []
    with h5py.File(Path(path).expanduser(), "r") as f:
        names = demos or sorted(f["data"], key=lambda k: int(k.removeprefix("demo_")))
        for name in names:
            group = f[f"data/{name}"]
            if not group.attrs.get("success", True):
                continue
            m = load_model(task, group.attrs["model_file"])
            d = mujoco.MjData(m)
            site = m.site(arm["site"]).id
            states, actions = group["states"][()], group["actions"][()]
            eef, objects, flags = [], {b: [] for b in bodies}, []
            for flat in states:
                d.qpos[:], d.qvel[:] = flat[1:1 + m.nq], flat[1 + m.nq:1 + m.nq + m.nv]
                mujoco.mj_forward(m, d)
                eef.append(_pose(d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()))
                for b in bodies:
                    i = m.body(b).id
                    objects[b].append(_pose(d.xpos[i].copy(), d.xmat[i].reshape(3, 3).copy()))
                flags.append(signals(m, d, spec["task"], d.site_xpos[site]))
            eef = np.stack(eef)
            arm_actions = np.clip(actions[:, :6], -1., 1.)
            targets = _pose(
                eef[:, :3, 3] + arm_actions[:, :3] * scale_pos,
                np.stack([_rotvec_to_mat(a[3:6] * scale_rot) @ e[:3, :3] for a, e in zip(arm_actions, eef)]),
            )
            ends, start = [], 0
            for subtask in subtasks[:-1]:
                signal = np.array([flag[subtask.signal] for flag in flags], int)
                rises = np.flatnonzero(np.diff(signal) > 0) + 1
                rises = rises[rises >= start]
                if not len(rises):
                    raise ValueError(f"{name}: subtask signal {subtask.signal!r} never turns on after step {start}")
                start = int(rises[0]) + 1
                ends.append(start)
            sources.append(Source(eef, targets, actions[:, 6:7], {b: np.stack(p) for b, p in objects.items()}, ends))
    return sources


class Plan(NamedTuple):
    """Per world: each subtask's gripper poses in its object's frame, gripper actions, noise, and length."""
    poses: np.ndarray  # (subtasks, steps, 4, 4)
    grips: np.ndarray  # (subtasks, steps, 1)
    noise: np.ndarray  # (subtasks,)
    lengths: np.ndarray  # (subtasks,)
    source: int


def plan(subtasks, sources, object_poses, rng):
    """
    MimicGen's choice of source demo and subtask boundaries for one generated demo starting with objects at
    `object_poses` (body -> 4 x 4 pose).
    """
    first = subtasks[0]
    if first.nearest and first.body is not None:
        current = object_poses[first.body]
        dists = []
        for s in sources:
            src = s.objects[first.body][0]
            delta = src[:3, :3] @ current[:3, :3].T
            angle = np.arccos(np.clip((np.trace(delta) - 1.) / 2., -1., 1.))
            dists.append(np.linalg.norm(src[:3, 3] - current[:3, 3]) + angle)
        nearest = np.argsort(dists)[:min(first.nearest, len(sources))]
        index = int(nearest[rng.integers(len(nearest))])
    else:
        index = int(rng.integers(len(sources)))
    source = sources[index]
    ends = [e + int(rng.integers(s.offsets[0], s.offsets[1] + 1)) for e, s in zip(source.ends, subtasks)]
    bounds = list(zip([0] + ends, ends + [len(source.grips)]))
    poses, grips = [], []
    for k, (subtask, (start, end)) in enumerate(zip(subtasks, bounds)):
        targets = source.targets[start:end]
        segment = np.concatenate([source.eef[start:start + 1], targets]) if k == 0 else targets
        # gripper actions shift one step later, as MimicGen pairs them with the target poses
        g = np.concatenate([source.grips[start:start + 1], source.grips[start:end]])[:len(segment)]
        if subtask.body is not None:
            segment = np.linalg.inv(source.objects[subtask.body][start]) @ segment
        poses.append(segment)
        grips.append(g)
    lengths = np.array([len(p) for p in poses])
    return Plan(poses, grips, np.array([s.noise for s in subtasks]), lengths, index)


def _stack_plans(plans):
    """Plans padded to a common number of steps per subtask and stacked over worlds."""
    longest = max(int(p.lengths.max()) for p in plans)
    pad = lambda x: np.concatenate([x, np.repeat(x[-1:], longest - len(x), 0)])
    poses = np.stack([np.stack([pad(s) for s in p.poses]) for p in plans]).astype(np.float32)
    grips = np.stack([np.stack([pad(s) for s in p.grips]) for p in plans]).astype(np.float32)
    return poses, grips, np.stack([p.noise for p in plans]).astype(np.float32), np.stack([p.lengths for p in plans])


def _rotvec(R):
    """Axis-angle vector of rotation matrix `R`."""
    q = mat2quat(R)
    w = jnp.clip(q[3], -1., 1.)
    s = jnp.sqrt(jnp.maximum(1. - w * w, 0.))
    angle = 2. * jnp.arccos(w)
    return jnp.where(s > 1e-8, q[:3] / jnp.where(s > 1e-8, s, 1.) * angle, jnp.zeros(3))


class Generator:
    """Jitted generation of one demo per world of the POMDP of a MimicGen task variant."""
    def __init__(self, task, worlds, model_xml=None):
        self.task, self.worlds = task, worlds
        self.subtasks = SUBTASKS[_family(task)]
        self.env = env = RobomimicPOMDP(
            task, model_xml=model_xml, reward="sparse", max_worlds=worlds, terminate_on_success=False,
        )
        arm_spec = task_spec(task)["arms"][0]
        self.scale_pos = float(arm_spec["arm_scale"]["output_max"][0])
        self.scale_rot = float(arm_spec["arm_scale"]["output_max"][3])
        self.bodies = [env.model.body(s.body).id if s.body is not None else -1 for s in self.subtasks]
        self.interp = np.array([s.interpolation for s in self.subtasks])
        arm = env.arms[0]

        def eef_pose(d):
            return d.site_xpos[arm.site], d.site_xmat[arm.site].reshape(3, 3)

        def frame(d, k):
            body = jnp.asarray(self.bodies)[k]
            rot = jnp.where(body >= 0, d.xmat[jnp.maximum(body, 0)].reshape(3, 3), jnp.eye(3))
            pos = jnp.where(body >= 0, d.xpos[jnp.maximum(body, 0)], jnp.zeros(3))
            return pos, rot

        def apply(frame_pos, frame_rot, pose):
            return frame_rot @ pose[:3, 3] + frame_pos, frame_rot @ pose[:3, :3]

        def action(d, pos, rot, grip, noise, key):
            """MimicGen's `target_pose_to_action`, with its noise."""
            cur_pos, cur_rot = eef_pose(d)
            dpos = jnp.clip((pos - cur_pos) / self.scale_pos, -1., 1.)
            drot = jnp.clip(_rotvec(rot @ cur_rot.T) / self.scale_rot, -1., 1.)
            arm_action = jnp.clip(jnp.concatenate([dpos, drot]) + noise * jax.random.normal(key, (6,)), -1., 1.)
            return jnp.concatenate([arm_action, grip])

        interp = jnp.asarray(self.interp)

        def step(carry, key):
            state, k, j, frame_pos, frame_rot, start_pos, start_rot, done, success, poses, grips, noise, lengths = carry
            d = state.data
            k_ = jnp.minimum(k, len(self.subtasks) - 1)
            entering = j == 0
            new_pos, new_rot = frame(d, k_)
            frame_pos = jnp.where(entering, new_pos, frame_pos)
            frame_rot = jnp.where(entering, new_rot, frame_rot)
            n_interp = interp[k_]
            first_pos, first_rot = apply(frame_pos, frame_rot, poses[k_, 0])
            # interpolation from the previous segment's last target pose (the gripper's pose for the first)
            tau = (j + 1).astype(jnp.float32) / (n_interp + 1).astype(jnp.float32)
            delta = _rotvec(first_rot @ start_rot.T)
            angle = jnp.linalg.norm(delta)
            interp_rot = jnp.where(angle < 0.05, first_rot, axisangle2mat(delta * tau) @ start_rot)
            interp_pos = start_pos + tau * (first_pos - start_pos)
            index = jnp.clip(j - n_interp, 0, poses.shape[1] - 1)
            seg_pos, seg_rot = apply(frame_pos, frame_rot, poses[k_, index])
            interpolating = j < n_interp
            pos = jnp.where(interpolating, interp_pos, seg_pos)
            rot = jnp.where(interpolating, interp_rot, seg_rot)
            grip = jnp.where(interpolating, grips[k_, 0], grips[k_, index])
            sigma = jnp.where(interpolating, 0., noise[k_])
            a = action(d, pos, rot, grip, sigma, key)
            flat = jnp.concatenate([d.time[None], d.qpos, d.qvel])
            next_state = env.step(None, state, a)
            reward = env.reward(None, state, a, next_state)
            success = success | (env.success(next_state) & ~done)
            # advance through the segment; its last target pose starts the next segment's interpolation
            j = j + 1
            finished = j >= n_interp + lengths[k_]
            last_pos, last_rot = apply(frame_pos, frame_rot, poses[k_, lengths[k_] - 1])
            start_pos = jnp.where(finished, last_pos, start_pos)
            start_rot = jnp.where(finished, last_rot, start_rot)
            k = jnp.where(finished, k + 1, k)
            j = jnp.where(finished, 0, j)
            recorded = (flat, a, reward, done)
            done = done | (k >= len(self.subtasks))
            return (next_state, k, j, frame_pos, frame_rot, start_pos, start_rot, done, success,
                    poses, grips, noise, lengths), recorded

        def generate(state, poses, grips, noise, lengths, keys):
            start_pos, start_rot = eef_pose(state.data)
            carry = (state, jnp.int32(0), jnp.int32(0), jnp.zeros(3), jnp.eye(3), start_pos, start_rot,
                     jnp.bool_(False), jnp.bool_(False), poses, grips, noise, lengths)
            carry, recorded = jax.lax.scan(step, carry, keys)
            return carry[8], recorded

        self._generate = jax.jit(jax.vmap(generate))
        self._reset = jax.jit(jax.vmap(env.reset_at))
        self._from_flat = jax.jit(jax.vmap(env.state_from_flat))
        self._objects = jax.jit(jax.vmap(lambda s: {
            b: (s.data.xpos[env.model.body(b).id], s.data.xmat[env.model.body(b).id].reshape(3, 3))
            for b in {s_.body for s_ in self.subtasks if s_.body is not None}
        }))

    def __call__(self, sources, resets, rng, starts=None):
        """
        One generated demo per reset index of `resets` (at most `worlds`), each a dict of states, actions, rewards,
        dones, success, the source demo, and the reset; with `starts`, one per flattened state of `starts`
        instead, simulated in the generator's model, and `resets` only labels them.
        """
        n = len(resets)
        resets = np.concatenate([resets, np.repeat(resets[-1:], self.worlds - n, axis=0)])
        if starts is None:
            states = self._reset(jnp.asarray(resets))
        else:
            starts = np.concatenate([starts, np.repeat(starts[-1:], self.worlds - n, axis=0)])
            states = self._from_flat(jnp.asarray(starts, jnp.float32))
        objects = jax.device_get(self._objects(states))
        plans = []
        for w in range(self.worlds):
            poses = {b: _pose(np.asarray(p[w]), np.asarray(r[w])) for b, (p, r) in objects.items()}
            plans.append(plan(self.subtasks, sources, poses, rng))
        poses, grips, noise, lengths = _stack_plans(plans)
        total = (lengths + self.interp).sum(axis=1)
        keys = jax.random.split(jax.random.key(int(rng.integers(2 ** 31))), self.worlds * int(total.max()))
        keys = keys.reshape(self.worlds, int(total.max()))
        success, (flats, actions, rewards, _) = self._generate(
            states, jnp.asarray(poses), jnp.asarray(grips), jnp.asarray(noise), jnp.asarray(lengths), keys,
        )
        success, flats, actions, rewards = (np.asarray(x) for x in (success, flats, actions, rewards))
        return [
            dict(states=flats[w, :total[w]].astype(np.float64), actions=actions[w, :total[w]].astype(np.float64),
                 rewards=rewards[w, :total[w]].astype(np.float64), success=bool(success[w]),
                 source=plans[w].source, reset=resets[w])
            for w in range(n)
        ]


def _model_file(task, fixtures, names):
    """The task's model xml with each body of `names` ({field: body names}) at its row of `fixtures` ({field: rows})."""
    with zipfile.ZipFile(_ASSETS / "models.zip") as z:
        root = ET.fromstring(z.read(f"models/{task}.xml").decode())
    bodies = {b.get("name"): b for b in root.iter("body")}
    for field, rows in fixtures.items():
        for name, row in zip(names[field], rows):
            body = bodies[name]
            body.set(field.removeprefix("body_"), " ".join(f"{x:.9g}" for x in row))
            if field == "body_quat":
                for attribute in ("euler", "axisangle", "xyaxes", "zaxis"):
                    body.attrib.pop(attribute, None)
    return ET.tostring(root, encoding="unicode")


def generate_dataset(task, source, output, demos=1000, worlds=1024, seed=0, max_rounds=20):
    """
    Generates `demos` successful demos of MimicGen task variant `task` from transferred source dataset `source` into
    hdf5 file `output`, each from a distinct reset of the task, in rounds of `worlds` attempts.
    """
    rng = np.random.default_rng(seed)
    sources = annotate(task, source)
    print(f"[{task}] {len(sources)} source demos, subtask ends {[s.ends for s in sources]}", flush=True)
    generator = Generator(task, worlds)
    env = generator.env
    resets_total = env._reset_qpos.shape[0]
    order = rng.permutation(resets_total)
    names = {f: [env.model.body(int(i)).name for i in index] for f, index in env._fixture_index.items()}
    env_name = next(k for k, v in TASK_OF_ENV_NAME.items() if v == task)
    with h5py.File(Path(source).expanduser(), "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    env_args["env_name"] = env_name
    out = h5py.File(Path(output).expanduser(), "w")
    data = out.create_group("data")
    data.attrs["env_args"] = json.dumps(env_args)
    kept, attempts, cursor = 0, 0, 0
    for round_ in range(max_rounds):
        if kept >= demos:
            break
        batch = np.array([order[(cursor + i) % resets_total] for i in range(worlds)])
        cursor += worlds
        for result in generator(sources, batch, rng):
            attempts += 1
            if not result["success"] or kept >= demos:
                continue
            group = data.create_group(f"demo_{kept}")
            for key in ("states", "actions", "rewards"):
                group.create_dataset(key, data=result[key])
            dones = np.zeros(len(result["actions"]), np.int64)
            dones[-1] = 1
            group.create_dataset("dones", data=dones)
            fixtures = {f: np.asarray(x[result["reset"]]) for f, x in zip(env._fixture_index, env._reset_fixtures)}
            group.attrs["model_file"] = _model_file(task, fixtures, names)
            group.attrs["num_samples"] = len(result["actions"])
            group.attrs["success"] = True
            group.attrs["source_demo"] = result["source"]
            group.attrs["reset"] = int(result["reset"])
            kept += 1
        out.flush()
        print(f"[{task}] round {round_ + 1}: {kept}/{demos} demos kept of {attempts} attempts "
              f"({kept / attempts:.0%} success)", flush=True)
    data.attrs["total"] = sum(g.attrs["num_samples"] for g in data.values())
    out.close()
    return kept, attempts
