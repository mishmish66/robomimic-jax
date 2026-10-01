"""
Transfer a robomimic dataset's demonstrations into the MuJoCo Warp POMDP.

Each demo is re-simulated from its first state by receding-horizon sampling: at every chunk, perturbations
of the recorded actions are rolled out in parallel worlds, and the one whose states stay closest to the
recorded states is kept. Demos are transferred in batches, each simulated in its own recorded model. The output has the input's demos, masks, and environment metadata, with each
demo's states, actions, rewards, and dones replaced by the simulated ones and its "success" attribute set.

Example:
    python transfer_demos.py --dataset ~/data/robomimic_v15/square/mh/demo_v15.hdf5 --output square_mh_warp.hdf5
"""
import argparse
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

import robomimic.pomdp
from robomimic.pomdp import TASK_OF_ENV_NAME, RobomimicPOMDP, load_model

# parked objects are those farther than this from the world origin
_PARKED_DISTANCE = 5.
# demo lengths are padded to multiples of this, bounding the number of compiled shapes
_LENGTH_BUCKET = 50
# scales of the tracking cost: object position (m), object orientation (rad), robot joints
_POSITION_SCALE, _ANGLE_SCALE, _JOINT_SCALE = 0.01, 0.1, 0.05
_JOINT_WEIGHT = 0.1


def _tracked_qpos(m, flat):
    """(object positions, object quaternions, robot joints) qpos indices; parked objects are left out."""
    positions, quaternions, free = [], [], set()
    for j in np.nonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]:
        adr = m.jnt_qposadr[j]
        free.update(range(adr, adr + 7))
        if np.linalg.norm(flat[1 + adr:4 + adr]) <= _PARKED_DISTANCE:
            positions.append(np.arange(adr, adr + 3))
            quaternions.append(np.arange(adr + 3, adr + 7))
    joints = np.array([i for i in range(m.nq) if i not in free])
    return np.concatenate(positions), np.stack(quaternions), joints


def _pad(x, n):
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), axis=0)])


class Transferred(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    success: bool
    last: np.ndarray  # flattened state after the last action


def _contact_success(m, flat, pairs):
    """Whether, at flattened state @flat of @m, every (geoms, geoms) pair of @pairs is in contact."""
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
    def __init__(self, env, first_state, samples, horizon, commit, iterations, graph_mode):
        self.env, self.samples, self.horizon, self.commit = env, samples, horizon, commit
        self._graph_mode = graph_mode
        self._tree = jax.tree_util.tree_structure(env._mx)
        nq = env._cpu_model.nq
        positions, quaternions, joints = _tracked_qpos(env._cpu_model, first_state)
        action_dim = env.action_space.shape[0]
        arm_dims = np.zeros(action_dim, np.float32)
        for arm in env._arms:
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
                mean = jax.lax.dynamic_slice_in_dim(actions, t, horizon)
                targets = jax.lax.dynamic_slice_in_dim(states, t + 1, horizon)
                mask = (t + jnp.arange(horizon) < length).astype(jnp.float32)
                for i in range(iterations):
                    key, sub = jax.random.split(key)
                    perturbation = noise * 0.5 ** i * jax.random.normal(sub, (samples, horizon, action_dim)) * arm_dims
                    candidates = jnp.clip(mean + perturbation.at[0].set(0.), -1., 1.)
                    flats, success, reward = jax.vmap(rollout, in_axes=(None, None, None, 0))(flat, grips[t], q0, candidates)
                    costs = cost(flats, targets, mask)
                    best = jnp.argmin(jnp.where(jnp.isnan(costs), jnp.inf, costs))
                    mean = candidates[best]
                kept = (flats[best, :commit], mean[:commit], reward[best, :commit], success[best, :commit])
                return (flats[best, commit - 1], key), kept
            chunks = (actions.shape[0] - horizon) // commit
            _, outputs = jax.lax.scan(chunk, (states[0], key), jnp.arange(chunks))
            return tuple(x.reshape(-1, *x.shape[2:]) for x in outputs)

        self._transfer = jax.jit(jax.vmap(transfer, in_axes=(0, 0, 0, 0, 0, 0, 0, None)))

    def _grips(self, actions):
        """Gripper controller state before each action."""
        grips = [np.zeros((len(self.env._arms), 2), np.float32)]
        for action in actions:
            action = jnp.asarray(action, jnp.float32)
            grips.append(np.stack([np.asarray(arm.step_grip(g, action)) for arm, g in zip(self.env._arms, grips[-1])]))
        return np.stack(grips)

    def model(self, model_xml):
        """MJX model of robosuite model xml @model_xml, with the structure of the transfer's own."""
        env = self.env
        m = env._warp_model(load_model(env._task_name, model_xml), np.asarray(env._reset_qpos[0]))
        mx = mjx.put_model(m, impl="warp", graph_mode=self._graph_mode)
        return jax.tree_util.tree_unflatten(self._tree, jax.tree_util.tree_leaves(mx))

    def __call__(self, demos, models, key, noise):
        """
        Simulated (states, actions, rewards, success) of each demo in @demos, a list of (states, actions) recorded
        in the MJX models @models, with arm action perturbations of standard deviation @noise.
        """
        stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *models)
        steps = -(-max(len(a) for _, a in demos) // _LENGTH_BUCKET) * _LENGTH_BUCKET
        padded = steps + self.horizon
        states = np.stack([_pad(s, padded + 1) for s, _ in demos]).astype(np.float32)
        actions = np.stack([_pad(a, padded) for _, a in demos]).astype(np.float32)
        grips = np.stack([_pad(self._grips(a), padded + 1) for _, a in demos])
        q0 = tuple(np.stack([s[0][1:][arm.qpos] for s, _ in demos]).astype(np.float32) for arm in self.env._arms)
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
    """Writes transferred @demo; per-step annotations of the recorded demo carry over unchanged."""
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
    """Combine shard files of one transfer into @output, with demos in the source's order."""
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


def main(args):
    if args.merge:
        return merge(Path(args.output).expanduser(), args.merge)
    source = h5py.File(Path(args.dataset).expanduser(), "r")
    env_args = json.loads(source["data"].attrs["env_args"])
    task = TASK_OF_ENV_NAME[env_args["env_name"]]
    names = sorted(source["data"], key=lambda k: int(k.removeprefix("demo_")))
    shard, shards = (int(x) for x in args.shard.split("/"))
    names = names[shard::shards]
    if args.limit:
        names = names[:args.limit]
    models = {name: source[f"data/{name}"].attrs["model_file"] for name in names}
    batch = args.batch
    names.sort(key=lambda n: len(source[f"data/{n}/actions"]))
    contact_pairs = None
    if task in _CONTACT_SUCCESS:
        spec = json.loads((Path(robomimic.pomdp.__file__).parent / "data" / "tasks.json").read_text())[task]["task"]
        contact_pairs = [(spec[a], spec[b]) for a, b in _CONTACT_SUCCESS[task]]

    output = Path(args.output).expanduser()
    if shards > 1:
        output = output.with_suffix(f".shard{shard}of{shards}.hdf5")
    out = h5py.File(output, "w")
    out.create_group("data")
    for k, v in source["data"].attrs.items():
        out["data"].attrs[k] = v
    if shard == 0 and "mask" in source:
        source.copy(source["mask"], out, "mask")

    print(f"[{task}] transferring {len(names)} demos from {args.dataset} (shard {shard}/{shards}, batch {batch}) -> {output}", flush=True)
    assert len(args.samples) == len(args.iterations), "--samples and --iterations give one value per stage"
    stages = [
        Transfer(
            RobomimicPOMDP(task, model_xml=models[names[0]], max_worlds=batch * samples),
            source[f"data/{names[0]}/states"][0], samples, args.horizon, args.commit, iterations,
            wp.JaxCallableGraphMode.WARP,
        )
        for samples, iterations in zip(args.samples, args.iterations)
    ]
    # each stage once, then the last stage again at growing noise
    attempts = [(stage, args.noise) for stage in stages[:-1]]
    attempts += [(stages[-1], args.noise * 1.5 ** k) for k in range(args.retries + 1)]
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

    for path in args.reuse or []:
        with h5py.File(Path(path).expanduser(), "r") as prior:
            for name in names:
                if name not in written and name in prior["data"] and prior[f"data/{name}"].attrs["success"]:
                    prior.copy(prior[f"data/{name}"], out["data"], name)
                    written.add(name)
    print(f"[{task}] reused {len(written)} successful demos", flush=True)

    start, best = time.time(), {}
    pending = [n for n in names if n not in written]
    for attempt, (transfer, noise) in enumerate(attempts):
        for b in range(0, len(pending), batch):
            group = pending[b:b + batch]
            padded = group + [group[-1]] * (batch - len(group))
            recorded = {n: (source[f"data/{n}/states"][()], source[f"data/{n}/actions"][()]) for n in group}
            mjx_models = {x: mjx_models.get(x) or stages[0].model(x) for x in {models[n] for n in group}}
            key = jax.random.fold_in(jax.random.key(args.seed + b), attempt)
            demos = transfer([recorded[n] for n in padded], [mjx_models[models[n]] for n in padded], key, noise)
            for name, demo in zip(group, demos):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="robomimic hdf5 dataset with states and actions")
    parser.add_argument("--output", required=True, help="hdf5 file for the transferred demos")
    parser.add_argument("--samples", type=int, nargs="+", default=[32, 256],
                        help="action sequences sampled per chunk, at each stage of search")
    parser.add_argument("--horizon", type=int, default=10, help="steps each sampled sequence is rolled out")
    parser.add_argument("--commit", type=int, default=5, help="steps of the best sequence kept per chunk")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2],
                        help="sampling rounds per chunk, each at half the noise, at each stage of search")
    parser.add_argument("--noise", type=float, default=0.1, help="standard deviation of the arm action perturbation")
    parser.add_argument("--batch", type=int, default=8, help="demos transferred together when they share a model")
    parser.add_argument("--shard", default="0/1", help="i/n: transfer every n-th demo starting at the i-th")
    parser.add_argument("--limit", type=int, default=0, help="transfer at most this many demos")
    parser.add_argument("--retries", type=int, default=3,
                        help="reruns of unsuccessful demos at the last stage, each at 1.5 times the noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge", nargs="+", help="shard files to combine into --output instead of transferring")
    parser.add_argument("--reuse", nargs="+", help="earlier outputs whose successful demos are copied, not transferred")
    main(parser.parse_args())
