"""
Transfer of recorded demonstrations into the MuJoCo Warp POMDP.

Each demo is re-simulated from its first state by receding-horizon sampling: at every chunk, perturbations
of the recorded actions are rolled out in parallel worlds, and the one whose states stay closest to the
recorded states is kept, with chunks that still drift searched again at growing noise. Demos are
transferred in batches, each simulated in its own recorded model.
"""
import json
import time
from pathlib import Path
from typing import NamedTuple

import h5py
import mujoco
import numpy as np

import jax
import jax.numpy as jnp
import warp as wp
from mujoco import mjx

from robomimic.env import TASK_OF_ENV_NAME, RobomimicPOMDP, load_model, parked_joints, task_spec

# demo lengths are padded to multiples of this, bounding the number of compiled shapes
_LENGTH_BUCKET = 50
# scales of the tracking cost: object position (m), object orientation (rad), robot joints
_POSITION_SCALE, _ANGLE_SCALE, _JOINT_SCALE = 0.01, 0.1, 0.05
_JOINT_WEIGHT = 0.1


def _tracked_qpos(m, flat):
    """(object positions, object quaternions, robot joints) qpos indices; parked objects are left out."""
    free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    tracked = np.setdiff1d(free, parked_joints(m, flat[1:1 + m.nq]))
    positions = np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + 3) for j in tracked])
    quaternions = np.stack([np.arange(m.jnt_qposadr[j] + 3, m.jnt_qposadr[j] + 7) for j in tracked])
    joints = np.setdiff1d(np.arange(m.nq), np.concatenate([np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + 7) for j in free]))
    return positions, quaternions, joints


def _pad(x, n):
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), axis=0)])


class Transferred(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    success: bool
    last: np.ndarray  # flattened state after the last action


def _contact_success(m, flat, pairs):
    """Whether, at flattened state `flat` of `m`, every (geoms, geoms) pair of `pairs` is in contact."""
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = flat[1:1 + m.nq], flat[1 + m.nq:1 + m.nq + m.nv]
    mujoco.mj_forward(m, d)
    touching = {frozenset((c.geom1, c.geom2)) for c in d.contact[:d.ncon]}
    return all(
        any(frozenset((m.geom(a).id, m.geom(b).id)) in touching for a in first for b in second)
        for first, second in pairs
    )


# tasks whose success is checked on each demo's own model, by MuJoCo contacts between these geom groups
_CONTACT_SUCCESS = {"transport": (("target_bin_geoms", "payload_geoms"), ("trash_bin_geoms", "trash_geoms"))}


class Transfer:
    """Jitted transfer of batches of demos, each in its own model of the same task."""
    def __init__(self, task, env, first_state, samples, horizon, commit, iterations, graph_mode, refine=0,
                 tolerance=1.):
        self.task, self.env, self.samples, self.horizon, self.commit = task, env, samples, horizon, commit
        self.batch = env.max_worlds // samples
        self._graph_mode = graph_mode
        self._tree = jax.tree_util.tree_structure(env._mx)
        nq = env.model.nq
        positions, quaternions, joints = _tracked_qpos(env.model, first_state)
        action_dim = env.action_space.shape[0]
        arm_dims = np.zeros(action_dim, np.float32)
        for arm in env.arms:
            arm_dims[arm.arm_action] = 1.

        def rollout(flat, grip, q0, actions):
            def step(state, action):
                state = env.step(None, state, action)
                d = state.data
                flat = jnp.concatenate([d.time[None], d.qpos, d.qvel])
                return state, (flat, env.success(state), env.reward(None, None, None, state))
            start = env.state_from_flat(flat)._replace(grip=grip, q0=q0)
            return jax.lax.scan(step, start, actions)[1]

        def cost(flats, targets, mask):
            q, target = flats[..., 1:1 + nq], targets[:, 1:1 + nq]
            position = jnp.sum((q[..., positions] - target[:, positions]) ** 2, -1) / _POSITION_SCALE ** 2
            alignment = jnp.abs(jnp.sum(q[..., quaternions] * target[:, quaternions], -1)).clip(0., 1.)
            angle = jnp.sum((2. * jnp.arccos(alignment)) ** 2, -1) / _ANGLE_SCALE ** 2
            joint = _JOINT_WEIGHT * jnp.sum((q[..., joints] - target[:, joints]) ** 2, -1) / _JOINT_SCALE ** 2
            return jnp.sum((position + angle + joint) * mask, -1)

        def transfer(model, states, actions, grips, q0, length, key, noise):
            simulated, env._mx = env._mx, model
            try:
                return _transfer(states, actions, grips, q0, length, key, noise)
            finally:
                env._mx = simulated

        def _transfer(states, actions, grips, q0, length, key, noise):
            def chunk(carry, c):
                flat, key = carry
                t = c * commit
                targets = jax.lax.dynamic_slice_in_dim(states, t + 1, horizon)
                mask = (t + jnp.arange(horizon) < length).astype(jnp.float32)

                def search(key, mean, scale):
                    """Best of `samples` perturbations of `mean`, one of which is `mean` itself."""
                    key, sub = jax.random.split(key)
                    perturbation = scale * jax.random.normal(sub, (samples, horizon, action_dim)) * arm_dims
                    candidates = jnp.clip(mean + perturbation.at[0].set(0.), -1., 1.)
                    flats, success, reward = jax.vmap(rollout, in_axes=(None, None, None, 0))(flat, grips[t], q0, candidates)
                    costs = cost(flats, targets, mask)
                    best = jnp.argmin(jnp.where(jnp.isnan(costs), jnp.inf, costs))
                    return key, (costs[best], candidates[best], flats[best], success[best], reward[best])

                key, best = search(key, jax.lax.dynamic_slice_in_dim(actions, t, horizon), noise)
                for i in range(1, iterations):
                    key, best = search(key, best[1], noise * 0.5 ** i)

                # chunks that still drift are searched again at growing noise around the best so far
                def drifting(loop):
                    r, _, best = loop
                    return (r < refine) & (best[0] > tolerance * mask.sum())

                def again(loop):
                    r, key, best = loop
                    key, best = search(key, best[1], noise * 1.5 ** (r + 1))
                    return r + 1, key, best

                _, key, (_, chosen, flats, success, reward) = jax.lax.while_loop(drifting, again, (0, key, best))
                kept = (flats[:commit], chosen[:commit], reward[:commit], success[:commit])
                return (flats[commit - 1], key), kept
            chunks = (actions.shape[0] - horizon) // commit
            _, outputs = jax.lax.scan(chunk, (states[0], key), jnp.arange(chunks))
            return tuple(x.reshape(-1, *x.shape[2:]) for x in outputs)

        self._transfer = jax.jit(jax.vmap(transfer, in_axes=(0, 0, 0, 0, 0, 0, 0, None)))

    def _grips(self, actions):
        """Gripper controller state before each action."""
        grips = [np.zeros((len(self.env.arms), 2), np.float32)]
        for action in actions:
            action = jnp.asarray(action, jnp.float32)
            grips.append(np.stack([np.asarray(arm.step_grip(g, action)) for arm, g in zip(self.env.arms, grips[-1])]))
        return np.stack(grips)

    def model(self, model_xml):
        """MJX model of robosuite model xml `model_xml`, with the structure of the transfer's own."""
        env = self.env
        m = env.simulated_model(model_xml)
        mx = mjx.put_model(m, impl="warp", graph_mode=self._graph_mode).tree_replace({"opt._impl.warn_overflow": 0})
        return jax.tree_util.tree_unflatten(self._tree, jax.tree_util.tree_leaves(mx))

    def __call__(self, demos, models, key, noise):
        """
        Simulated (states, actions, rewards, success) of each demo in `demos`, a list of (states, actions) recorded
        in the MJX models `models`, with arm action perturbations of standard deviation `noise`.
        """
        stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *models)
        steps = -(-max(len(a) for _, a in demos) // _LENGTH_BUCKET) * _LENGTH_BUCKET
        padded = steps + self.horizon
        states = np.stack([_pad(s, padded + 1) for s, _ in demos]).astype(np.float32)
        actions = np.stack([_pad(a, padded) for _, a in demos]).astype(np.float32)
        grips = np.stack([_pad(self._grips(a), padded + 1) for _, a in demos])
        q0 = tuple(np.stack([s[0][1:][arm.qpos] for s, _ in demos]).astype(np.float32) for arm in self.env.arms)
        lengths = np.array([len(a) for _, a in demos])
        keys = jax.random.split(key, len(demos))
        outputs = self._transfer(stacked, states, actions, grips, q0, lengths, keys, jnp.float32(noise))
        flats, chosen, rewards, success = (np.asarray(x) for x in outputs)
        return [
            Transferred(
                np.concatenate([recorded[:1], flats[i, :n - 1]]).astype(np.float64), chosen[i, :n].astype(np.float64),
                rewards[i, :n].astype(np.float64), bool(success[i, :n].any()), flats[i, n - 1].astype(np.float64),
            )
            for i, ((recorded, _), n) in enumerate(zip(demos, lengths))
        ]


def _write_demo(group, recorded, demo):
    """Writes transferred `demo`; per-step annotations of the recorded demo carry over unchanged."""
    for k, v in recorded.attrs.items():
        group.attrs[k] = v
    group.attrs["num_samples"] = len(demo.actions)
    group.attrs["success"] = demo.success
    dones = np.zeros(len(demo.actions), dtype=np.int64)
    dones[-1] = 1
    simulated = dict(states=demo.states, actions=demo.actions, rewards=demo.rewards, dones=dones)
    for name, value in simulated.items():
        group.create_dataset(name, data=value)
    for name, value in recorded.items():
        if name not in simulated and isinstance(value, h5py.Dataset):
            recorded.copy(value, group, name)


def merge(output, shards):
    """Combines the shard files `shards` of one transfer into `output`."""
    with h5py.File(output, "w") as out:
        for i, path in enumerate(shards):
            with h5py.File(path, "r") as shard:
                if i == 0:
                    out.create_group("data")
                    for k, v in shard["data"].attrs.items():
                        out["data"].attrs[k] = v
                for name in shard["data"]:
                    shard.copy(shard[f"data/{name}"], out["data"], name)
                if "mask" in shard:
                    shard.copy(shard["mask"], out, "mask")
        out["data"].attrs["total"] = sum(g.attrs["num_samples"] for g in out["data"].values())
        failures = [n for n, g in out["data"].items() if not g.attrs["success"]]
        print(f"merged {len(out['data'])} demos into {output}; unsuccessful: {failures}")


def transfer_dataset(dataset, output, *, samples=(32, 256), iterations=(1, 2), horizon=10, commit=5, noise=0.1,
                     refine=0, tolerance=1., retries=3, worlds=4096, shard="0/1", demos=None, limit=0, reuse=(), seed=0):
    """
    Transfers the demos of robomimic dataset `dataset` into hdf5 file `output`, which gets the dataset's demos,
    masks, and environment metadata, with each demo's states, actions, rewards, and dones simulated and its
    "success" attribute set. The search runs once at each stage (`samples`, `iterations`), then `retries` more
    times at the last stage with 1.5 times the noise each. Successful demos of the earlier outputs `reuse` are
    copied instead of transferred. With `shard` "i/n", every n-th demo from the i-th goes to a file of its own.
    Each stage transfers as many demos together as fit in `worlds` worlds.
    """
    source = h5py.File(Path(dataset).expanduser(), "r")
    env_args = json.loads(source["data"].attrs["env_args"])
    task = TASK_OF_ENV_NAME[env_args["env_name"]]
    names = sorted(source["data"], key=lambda k: int(k.removeprefix("demo_")))
    shard, shards = (int(x) for x in shard.split("/"))
    names = names[shard::shards]
    if demos:
        names = [n for n in names if n in demos]
    if limit:
        names = names[:limit]
    models = {name: source[f"data/{name}"].attrs["model_file"] for name in names}
    names.sort(key=lambda n: len(source[f"data/{n}/actions"]))
    contact_pairs = None
    if task in _CONTACT_SUCCESS:
        spec = task_spec(task)["task"]
        contact_pairs = [(spec[a], spec[b]) for a, b in _CONTACT_SUCCESS[task]]

    output = Path(output).expanduser()
    if shards > 1:
        output = output.with_suffix(f".shard{shard}of{shards}.hdf5")
    out = h5py.File(output, "w")
    out.create_group("data")
    for k, v in source["data"].attrs.items():
        out["data"].attrs[k] = v
    if shard == 0 and "mask" in source:
        source.copy(source["mask"], out, "mask")

    print(f"[{task}] transferring {len(names)} demos from {dataset} (shard {shard}/{shards}) -> {output}", flush=True)
    assert len(samples) == len(iterations), "`samples` and `iterations` give one value per stage"
    stages = [
        Transfer(
            task, RobomimicPOMDP(task, model_xml=models[names[0]], max_worlds=max(1, worlds // stage_samples) * stage_samples),
            source[f"data/{names[0]}/states"][0], stage_samples, horizon, commit, stage_iterations,
            wp.JaxCallableGraphMode.WARP, refine, tolerance,
        )
        for stage_samples, stage_iterations in zip(samples, iterations)
    ]
    # each stage once, then the last stage again at growing noise
    attempts = [(stage, noise) for stage in stages[:-1]]
    attempts += [(stages[-1], noise * 1.5 ** k) for k in range(retries + 1)]
    mjx_models, cpu_models = {}, {}

    def transferred(name, demo):
        """@demo with its success and rewards judged on its own model where the task requires it."""
        if contact_pairs is None:
            return demo
        xml = models[name]
        m = cpu_models.setdefault(xml, load_model(task, xml))
        after = np.concatenate([demo.states[1:], demo.last[None]])
        success = np.array([_contact_success(m, flat, contact_pairs) for flat in after])
        return demo._replace(success=bool(success.any()), rewards=success.astype(np.float64))

    written = set()

    def write(name, demo):
        _write_demo(out["data"].create_group(name), source[f"data/{name}"], demo)
        written.add(name)
        out.flush()

    for path in reuse:
        with h5py.File(Path(path).expanduser(), "r") as prior:
            for name in names:
                if name not in written and name in prior["data"] and prior[f"data/{name}"].attrs["success"]:
                    prior.copy(prior[f"data/{name}"], out["data"], name)
                    written.add(name)
    print(f"[{task}] reused {len(written)} successful demos", flush=True)

    start, best = time.time(), {}
    pending = [n for n in names if n not in written]
    for attempt, (transfer, attempt_noise) in enumerate(attempts):
        if not pending:
            break
        batch = min(transfer.batch, len(pending))
        for b in range(0, len(pending), batch):
            group = pending[b:b + batch]
            padded = group + [group[-1]] * (transfer.batch - len(group))
            recorded = {n: (source[f"data/{n}/states"][()], source[f"data/{n}/actions"][()]) for n in group}
            mjx_models = {x: mjx_models.get(x) or stages[0].model(x) for x in {models[n] for n in group}}
            key = jax.random.fold_in(jax.random.key(seed + b), attempt)
            results = transfer([recorded[n] for n in padded], [mjx_models[models[n]] for n in padded], key, attempt_noise)
            for name, demo in zip(group, results):
                demo = transferred(name, demo)
                if demo.success:
                    write(name, demo)
                    best.pop(name, None)
                elif name not in best:
                    best[name] = demo
            elapsed = time.time() - start
            print(f"[{task}] attempt {attempt + 1}/{len(attempts)}: {min(b + batch, len(pending))}/{len(pending)} demos | "
                  f"{len(written)}/{len(names)} successful | {elapsed / 60:.1f} min elapsed", flush=True)
        pending = [n for n in pending if n not in written]
        if not pending:
            break
    for name in pending:
        write(name, best[name])
    out["data"].attrs["total"] = sum(out[f"data/{n}"].attrs["num_samples"] for n in names)
    out.close()
    print(f"[{task}] done: {len(names) - len(pending)}/{len(names)} successful; unsuccessful: {pending}", flush=True)
