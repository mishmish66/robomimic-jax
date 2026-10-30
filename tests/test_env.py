"""
Tests of the MuJoCo Warp POMDPs against the robomimic demonstration datasets. The Lift test dataset (with
recorded camera images) is downloaded to tests/assets if missing; tests on the task datasets run when
$ROBOMIMIC_DATA/<task>/<ph|mh>/ holds demo_v15.hdf5 (raw) or low_dim_v15.hdf5 (with observations), and, for the
MimicGen tasks, when $MIMICGEN_DATA holds their generated dataset.
"""
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax_pomdps as pomdps
import mujoco
import warp as wp

from robomimic.data import generate, hdf5, registry
from robomimic.env import PIXELS, TASK_OF_ENV_NAME, TASKS, Pixels, RobomimicPOMDP, parked_joints, registry_name
from robomimic.env.geometry import ContactTest
from robomimic.env import render
from robomimic.env.render import CameraRenderer

NUM_WORLDS = 4
CAMERAS = ["agentview", "robot0_eye_in_hand"]
# robosuite's shaped rewards at the states of the first demos, recorded by robosuite_shaped_rewards.py
ROBOSUITE_SHAPED_REWARDS = json.loads((Path(__file__).parent / "robosuite_shaped_rewards.json").read_text())


@pytest.fixture(scope="session")
def lift_test_dataset():
    folder = Path(__file__).parent / "assets"
    path = folder / "test_v15.hdf5"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        registry.download_file_from_hf(
            repo_id=registry.HF_REPO_ID, filename="test/test_v15.hdf5", download_dir=folder, check_overwrite=False
        )
    return path


_SHARED = {"task": None, "envs": {}}


def _env(task, variant="", model_xml=None, max_worlds=NUM_WORLDS):
    """
    The environment of `task` under registry variant `variant`, or simulating robosuite model xml `model_xml`, shared by
    the tests of the task; the environments of the previous task are dropped.
    """
    if _SHARED["task"] != task:
        _SHARED["task"], _SHARED["envs"] = task, {}
        gc.collect()
    key = (variant, model_xml, max_worlds)
    if key not in _SHARED["envs"]:
        _SHARED["envs"][key] = (
            RobomimicPOMDP(task, model_xml=model_xml, max_worlds=max_worlds) if model_xml is not None
            else pomdps.make(f"robomimic.env:{registry_name(task)}{variant}", max_worlds=max_worlds)
        )
    return _SHARED["envs"][key]


def _jit(env, name, vmap=False):
    """`env`'s method `name`, jitted (and mapped over worlds with `vmap`) once per environment."""
    cache = env.__dict__.setdefault("_test_jits", {})
    if (name, vmap) not in cache:
        method = getattr(env, name)
        cache[name, vmap] = jax.jit(jax.vmap(method) if vmap else method)
    return cache[name, vmap]


def _task_dataset(task, filename="demo_v15.hdf5"):
    if task in registry.MIMICGEN_DATASETS:
        path = registry.MIMICGEN_DATA_DIR / registry.MIMICGEN_DATASETS[task]
        if path.exists():
            return path
        pytest.skip(f"no {path}")
    for dataset_type in ["ph", "mh"]:
        path = registry.DATA_DIR / task / dataset_type / filename
        if path.exists():
            return path
    pytest.skip(f"no {task} {filename} under {registry.DATA_DIR}")


def _demo(path, name):
    with h5py.File(path, "r") as f:
        demo = f["data/" + name]
        return dict(
            model=demo.attrs["model_file"],
            states=demo["states"][()],
            actions=demo["actions"][()],
            obs={k: demo["obs/" + k][()] for k in demo.get("obs", {})},
        )


def _recorded_states(env, flats):
    """States at a demo's recorded states `flats`, each with the task's memory of the states up to it."""
    def remember(memory, flat):
        memory = env._task.remember(env.state_from_flat(flat).data, memory)
        return memory, memory

    _, memory = jax.jit(lambda f: jax.lax.scan(remember, env._task.initial_memory(), f))(jnp.asarray(flats, jnp.float32))
    return _jit(env, "state_from_flat", vmap=True)(flats)._replace(memory=memory)


def _episode_states(env, demo, starts):
    """States at the demo's steps `starts` with the controller state the demo had there."""
    grip = jnp.zeros((len(env.arms), 2), jnp.float32)
    grips = []
    for t in range(max(starts) + 1):
        grips.append(grip)
        action = jnp.asarray(demo["actions"][t], jnp.float32)
        grip = jnp.stack([arm.step_grip(g, action) for arm, g in zip(env.arms, grip)])
    q0 = tuple(jnp.asarray(demo["states"][0][1:][arm.qpos], jnp.float32) for arm in env.arms)
    states = _jit(env, "state_from_flat", vmap=True)(demo["states"][starts])
    return states._replace(grip=jnp.stack([grips[t] for t in starts]), q0=tuple(jnp.broadcast_to(q, (len(starts),) + q.shape) for q in q0))


def _live_qpos(m, flat):
    """qpos indices of everything but the objects robosuite parks far outside the workspace at state `flat`."""
    parked = [np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + 7) for j in parked_joints(m, flat[1:1 + m.nq])]
    return np.setdiff1d(np.arange(m.nq), np.concatenate(parked) if parked else [])


@pytest.fixture(scope="module", params=list(TASKS))
def env(request):
    return _env(request.param)


def test_tasks_are_registered_pomdps(env):
    assert isinstance(env, pomdps.POMDP)
    assert env.action_space.shape == (7 * len(env.arms),)


def test_unknown_tasks_are_rejected():
    with pytest.raises(ValueError):
        RobomimicPOMDP("lift_real")


def test_tasks_register_under_dashed_names():
    assert registry_name("tool_hang") == "tool-hang"
    assert isinstance(pomdps.make("robomimic.env:tool-hang", max_worlds=1), RobomimicPOMDP)


@pytest.mark.parametrize("task", list(TASKS))
def test_names_without_a_reward_variant_are_dense_and_sparse_names_are_sparse(task):
    key = jax.random.key(0)
    rewards = {}
    for suffix in ["", "/sparse"]:
        env = _env(task, suffix)
        down = jnp.zeros(env.action_space.shape).at[np.array([arm.arm_action.start + 2 for arm in env.arms])].set(-1.)
        state = _jit(env, "reset")(key)
        next_state = _jit(env, "step")(key, state, down)
        rewards[suffix] = float(_jit(env, "reward")(key, state, down, next_state))
    assert rewards["/sparse"] == 0.
    assert rewards[""] != 0.


@pytest.mark.parametrize("task", list(TASKS))
def test_dense_rewards_along_a_demo_pay_one_for_success_plus_the_progress_it_makes(task):
    demo = _demo(_task_dataset(task), "demo_0")
    env = _env(task, model_xml=demo["model"], max_worlds=len(demo["states"]))
    states = _recorded_states(env, demo["states"])
    success = np.asarray(jax.vmap(env.success)(states))
    end = int(np.argmax(success))
    assert success[end] and not success[:end].any()
    from_flat = _jit(env, "state_from_flat", vmap=True)
    before, after = (
        from_flat(demo["states"][steps])._replace(memory=jax.tree.map(lambda x: x[steps], states.memory))
        for steps in (slice(0, end), slice(1, end + 1))
    )
    rewards = np.asarray(jax.jit(jax.vmap(lambda s, n: env.reward(None, s, None, n)))(before, after))
    first = _jit(env, "state_from_flat")(demo["states"][0])
    start = float(env._task.progress(first.data, *jax.tree.map(lambda x: x[0], states.memory)))
    np.testing.assert_allclose(rewards.sum(), 1. + (1. - start), atol=1e-4)
    assert rewards[-1] >= 1.
    assert np.all(np.abs(rewards[:-1]) <= 1.)


@pytest.mark.parametrize("task", list(TASKS))
def test_variants_observe_low_dim_proprioception_images_or_markov_state(task):
    name = registry_name(task)
    cameras, size = PIXELS[task]
    env = _env(task)
    low_dim = env.observation_space.spaces
    extra = set(jax.eval_shape(lambda s: env._task.extra_obs(s.data), jax.eval_shape(env.reset, jax.random.key(0))))
    proprio = [k for k in low_dim if k != "object" and k not in extra]
    assert proprio and all(k.startswith("robot") for k in proprio)
    proprio_size = sum(int(np.prod(low_dim[k].shape)) for k in proprio)
    pixels = pomdps.spaces.Image(size, size, 3 * len(cameras))
    prp = _env(task, "/prp").observation_space
    assert prp.shape == (proprio_size,)
    assert _env(task, "/pix").observation_space == pixels
    assert _env(task, "/pix-prp").observation_space == pomdps.spaces.Dict(
        {"pixels": pixels, "prp": prp}
    )
    mkv = _env(task, "/mkv").observation_space
    assert len(mkv.shape) == 1 and mkv.shape[0] > low_dim["object"].shape[0] + proprio_size
    for variant in ["", "/prp", "/pix", "/pix-prp", "/mkv"]:
        env = _env(task, variant)
        obs = _jit(env, "observe")(None, _jit(env, "reset")(jax.random.key(0)), None)
        assert bool(env.observation_space.contains(obs)), variant


def _from_mkv(env, obs, base):
    """State of `env` rebuilt from its "mkv" observation `obs`, with parked objects as in state `base`."""
    m = env.model
    qpos, qvel = np.asarray(base.data.qpos), np.asarray(base.data.qvel)
    live_qpos = _live_qpos(m, np.concatenate([[0.], qpos]))
    parked = parked_joints(m, qpos)
    live_qvel = np.setdiff1d(np.arange(m.nv), [m.jnt_dofadr[j] + np.arange(6) for j in parked])
    memory_sizes, fixture_sizes = [x.size for x in base.memory], [x.size for x in base.fixtures]
    sizes = [len(live_qpos), len(live_qvel), base.grip.size, *memory_sizes, *fixture_sizes, sum(len(q) for q in base.q0)]
    qpos_part, qvel_part, grip, *parts, q0_offset = jnp.split(obs[-sum(sizes):], np.cumsum(sizes)[:-1])
    memory, fixtures = parts[:len(memory_sizes)], parts[len(memory_sizes):]
    fixtures = tuple(part.reshape(x.shape) for part, x in zip(fixtures, base.fixtures))
    state = env.init_state(
        jnp.asarray(qpos).at[live_qpos].set(qpos_part), jnp.asarray(qvel).at[live_qvel].set(qvel_part), fixtures=fixtures,
    )
    offsets = jnp.split(q0_offset, len(base.q0))
    q0 = tuple(offset + state.data.qpos[arm.qpos] for arm, offset in zip(env.arms, offsets))
    memory = tuple(part.reshape(x.shape).astype(x.dtype) for part, x in zip(memory, base.memory))
    return state._replace(grip=grip.reshape(base.grip.shape), q0=q0, memory=memory)


@pytest.mark.parametrize("task", list(TASKS))
def test_mkv_observation_and_action_determine_the_next_observation_reward_and_done(task):
    """
    The state rebuilt from an "mkv" observation steps as the original does from the same solver warm start, within
    the spread of repeated steps of the original, which contact-rich scenes make nondeterministic.
    """
    env = _env(task, "/mkv")
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    step, observe = _jit(env, "step", vmap=True), _jit(env, "observe", vmap=True)
    reward, done = _jit(env, "reward", vmap=True), _jit(env, "done", vmap=True)
    sample = jax.vmap(env.action_space.sample)
    states = _jit(env, "reset", vmap=True)(keys)
    for i in range(3):
        states = step(keys, states, sample(jax.random.split(jax.random.key(i + 1), NUM_WORLDS)))
    base = _jit(env, "reset")(jax.random.key(7))
    rebuilt = jax.jit(jax.vmap(lambda o: _from_mkv(env, o, base)))(observe(keys, states, None))
    actions = sample(jax.random.split(jax.random.key(9), NUM_WORLDS))
    original = states._replace(data=states.data.replace(qacc_warmstart=rebuilt.data.qacc_warmstart))

    def outcome(start):
        nxt = step(keys, start, actions)
        return np.asarray(observe(keys, nxt, actions)), np.asarray(reward(keys, start, actions, nxt)), np.asarray(done(nxt))

    repeats = [outcome(original) for _ in range(4)]
    obs, rew, dones = outcome(rebuilt)
    for i, got in enumerate((obs, rew)):
        spread = max(np.abs(r[i] - repeats[0][i]).max() for r in repeats)
        np.testing.assert_allclose(got, repeats[0][i], atol=1e-4 + 2 * spread)
    assert np.all(np.any([dones == r[2] for r in repeats], axis=0))


@pytest.mark.parametrize("variant", ["", "/sparse", "/mkv", "/prp"])
def test_diverged_simulations_end_the_episode_taking_back_progress_with_finite_observations(variant):
    env = pomdps.make(f"robomimic.env:lift{variant}", max_worlds=1)
    key = jax.random.key(0)
    state = _jit(env, "reset")(key)
    diverged = state._replace(data=state.data.replace(qvel=state.data.qvel.at[0].set(jnp.nan)))
    assert not bool(env.diverged(state)) and bool(env.diverged(diverged))
    assert bool(_jit(env, "done")(diverged))
    progress = 0. if variant == "/sparse" else float(env._task._potential(state.data, state.memory))
    assert float(_jit(env, "reward")(key, state, None, diverged)) == pytest.approx(-progress)
    assert float(_jit(env, "reward")(key, diverged, None, diverged)) == 0.
    obs = _jit(env, "observe")(key, diverged, None)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(obs))


def test_unknown_observations_and_rewards_are_rejected():
    with pytest.raises(ValueError):
        RobomimicPOMDP("lift", observation="depth")
    with pytest.raises(ValueError):
        RobomimicPOMDP("lift", reward="shaped")


def test_batched_reset_step_observe_under_jit_and_vmap(env):
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    states = _jit(env, "reset", vmap=True)(keys)
    actions = jax.vmap(env.action_space.sample)(keys) * 0.3
    next_states = _jit(env, "step", vmap=True)(keys, states, actions)
    obs = _jit(env, "observe", vmap=True)(keys, next_states, actions)
    rewards = _jit(env, "reward", vmap=True)(keys, states, actions, next_states)
    dones = _jit(env, "done", vmap=True)(next_states)
    assert rewards.shape == (NUM_WORLDS,)
    assert dones.shape == (NUM_WORLDS,)
    assert not bool(jnp.any(jax.vmap(env.overflowed)(next_states)))
    for k, space in env.observation_space.spaces.items():
        assert obs[k].shape == (NUM_WORLDS,) + space.shape, k
        assert bool(jnp.all(jnp.isfinite(obs[k]))), k


def test_flat_state_round_trips(env):
    state = _jit(env, "reset")(jax.random.key(1))
    flat = env.flat_state(state)
    np.testing.assert_allclose(env.flat_state(_jit(env, "state_from_flat")(flat)), flat, atol=1e-6)


@pytest.mark.parametrize("task", list(TASKS))
def test_observations_match_dataset(task):
    """Observations at recorded states equal the recorded ones; robosuite recorded each state's observation
    right after the previous state's, which object poses relative to the gripper are cached from."""
    demo = _demo(_task_dataset(task, "low_dim_v15.hdf5"), "demo_0")
    steps = np.linspace(1, len(demo["states"]) - 1, NUM_WORLDS).astype(int)
    env = _env(task, model_xml=demo["model"], max_worlds=len(demo["states"]))
    init = _jit(env, "state_from_flat", vmap=True)
    previous = init(demo["states"][steps - 1])
    states = init(demo["states"][steps])._replace(cache=previous.cache)
    obs = _jit(env, "observe", vmap=True)(None, states, None)
    assert set(obs) == {k for k in demo["obs"] if not k.endswith("_image")}
    for k in obs:
        np.testing.assert_allclose(np.asarray(obs[k]), demo["obs"][k][steps], atol=1e-5, err_msg=k)


def test_camera_depth_matches_dataset_depth(lift_test_dataset):
    demo = _demo(lift_test_dataset, "demo_0")
    height, width = demo["obs"]["agentview_image"].shape[1:3]
    env = RobomimicPOMDP(
        "lift", model_xml=demo["model"], observation=Pixels(tuple(CAMERAS), height, width), max_worlds=NUM_WORLDS
    )
    steps = np.linspace(0, len(demo["states"]) - 1, NUM_WORLDS).astype(int)
    states = _jit(env, "state_from_flat", vmap=True)(demo["states"][steps])
    pixels = np.asarray(_jit(env, "observe", vmap=True)(None, states, None))
    renderer = CameraRenderer(env.model, CAMERAS, height, width, depth=True, nworld=NUM_WORLDS,
                              graph_mode=wp.JaxCallableGraphMode.WARP)
    rendered = jax.jit(jax.vmap(renderer))(states.data)
    for i, camera in enumerate(CAMERAS):
        assert pixels.dtype == np.uint8 and pixels[..., 3 * i:3 * i + 3].shape == demo["obs"][camera + "_image"][steps].shape
        depth, expected = np.asarray(rendered[camera + "_depth"]), demo["obs"][camera + "_depth"][steps]
        assert np.median(np.abs(depth - expected)) < 0.01, camera


def test_table_top_pixels_reconstruct_onto_its_top_face_and_show_its_texture():
    env = RobomimicPOMDP("square", observation=Pixels(tuple(CAMERAS), 84, 84), max_worlds=1)
    renderer = env._renderer
    d = _jit(env, "reset")(jax.random.key(0)).data
    color, depth, seg = renderer._scene(d)
    rc = renderer._scene.context.pytree()
    table = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    image = np.asarray(renderer(d)["agentview_image"])
    for i in range(len(CAMERAS)):
        front = render.mjx.get_segmentation(rc, i, seg)
        z = render.mjx.get_depth(rc, i, depth, renderer._far)[..., 0] * renderer._far
        local = np.asarray(renderer._textures.local_points(d, i, front, z))
        on = np.asarray(front) == table
        assert on.mean() > 0.3
        assert np.abs(local[on][:, 2] - env.model.geom_size[table][2]).max() < 1e-3
    on = np.asarray(render.mjx.get_segmentation(rc, 0, seg))[::render.SUPERSAMPLE, ::render.SUPERSAMPLE] == table
    assert image[on].astype(float).std(0).mean() > 8.


def test_pixel_observations_match_their_spaces_unbatched_and_mapped():
    env = RobomimicPOMDP("lift", observation=Pixels(tuple(CAMERAS), 48, 64, proprio=True), max_worlds=NUM_WORLDS)
    space = env.observation_space.spaces
    assert space["pixels"] == pomdps.spaces.Image(48, 64, 3 * len(CAMERAS))
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    mapped = _jit(env, "observe", vmap=True)(keys, _jit(env, "reset", vmap=True)(keys), None)
    single = _jit(env, "observe")(keys[1], _jit(env, "reset")(keys[1]), None)
    for k in ["pixels", "prp"]:
        assert mapped[k].shape == (NUM_WORLDS,) + space[k].shape, k
        np.testing.assert_array_equal(np.asarray(single[k]), np.asarray(mapped[k][1]), err_msg=k)
    assert int(jnp.ptp(mapped["pixels"])) > 0


def test_garbage_collection_pauses_while_graphs_are_captured(monkeypatch):
    begin, collecting = wp.capture_begin, []

    def capture_begin(*args, **kwargs):
        collecting.append(gc.isenabled())
        return begin(*args, **kwargs)

    monkeypatch.setattr(wp, "capture_begin", capture_begin)
    env = RobomimicPOMDP("lift", observation=Pixels(("agentview",), 32, 32), max_worlds=1)
    state = _jit(env, "step")(None, _jit(env, "reset")(jax.random.key(0)), jnp.zeros(7))
    jax.block_until_ready(_jit(env, "observe")(None, state, None))
    assert collecting and not any(collecting)
    assert gc.isenabled()


def test_warp_callables_keep_at_most_eight_captured_graphs():
    from warp._src.jax import ffi
    env = RobomimicPOMDP("lift", max_worlds=4)
    step = _jit(env, "step", vmap=True)
    keys = jax.random.split(jax.random.key(0), 4)
    states = _jit(env, "reset", vmap=True)(keys)
    for i in range(12):
        states = step(keys, states, jnp.full((4, 7), 0.1 * (i % 3) - 0.1))
    jax.block_until_ready(states)
    callables = list(ffi._FFI_CALLABLE_REGISTRY.values())
    assert callables
    assert all(c.graph_cache_max is not None and c.graph_cache_max <= 8 for c in callables)
    assert all(len(c.captures) <= 8 for c in callables)


def test_open_loop_replay_of_lift_demos_succeeds_and_tracks_the_cube(lift_test_dataset):
    for name in ["demo_0", "demo_1"]:
        demo = _demo(lift_test_dataset, name)
        env = RobomimicPOMDP("lift", model_xml=demo["model"], max_worlds=1)
        step = jax.jit(lambda s, a: env.step(None, s, a))
        state = env.state_from_flat(demo["states"][0])
        m = env.model
        cube = m.jnt_qposadr[m.body("cube_main").jntadr[0]] + np.arange(3)
        succeeded = False
        for t, action in enumerate(demo["actions"]):
            state = step(state, jnp.asarray(action, jnp.float32))
            succeeded |= bool(env.success(state))
            assert not bool(env.overflowed(state))
            if t + 1 < len(demo["states"]):
                np.testing.assert_allclose(np.asarray(state.data.qpos)[cube], demo["states"][t + 1][1:][cube], atol=0.02)
        assert succeeded, name


@pytest.mark.parametrize("task", list(TASKS))
def test_demos_end_in_success_and_start_without_it(task):
    path = _task_dataset(task)
    assert TASK_OF_ENV_NAME[hdf5.get_env_metadata_from_dataset(path)["env_name"]] == task
    demo = _demo(path, "demo_0")
    env = _env(task, model_xml=demo["model"], max_worlds=len(demo["states"]))
    success = np.asarray(jax.vmap(env.success)(_recorded_states(env, demo["states"])))
    first, last = success[0], success[-1]
    assert not first and last


@pytest.mark.parametrize("task", list(TASKS))
def test_one_step_transitions_match_dataset(task):
    """Each demo transition, simulated from its recorded state and controller state, lands on the next recorded state
    in the coordinates that take part in the task."""
    demo = _demo(_task_dataset(task), "demo_0")
    starts = np.arange(len(demo["actions"]) - 1)
    env = _env(task, model_xml=demo["model"], max_worlds=len(demo["states"]))
    states = _episode_states(env, demo, starts)
    next_states = _jit(env, "step", vmap=True)(None, states, jnp.asarray(demo["actions"][starts], jnp.float32))
    assert not bool(jnp.any(jax.vmap(env.overflowed)(next_states)))
    live = _live_qpos(env.model, demo["states"][0])
    error = np.abs(np.asarray(next_states.data.qpos)[:, live] - demo["states"][starts + 1][:, 1:][:, live])
    assert np.median(error.max(axis=1)) < 1e-3
    assert np.percentile(error.max(axis=1), 95) < 1e-2


@pytest.mark.parametrize("task", list(TASKS))
def test_shaped_rewards_match_robosuite(task):
    path = _task_dataset(task)
    for name, expected in ROBOSUITE_SHAPED_REWARDS[task].items():
        demo = _demo(path, name)
        env = _env(task, model_xml=demo["model"], max_worlds=len(demo["states"]))
        states = _recorded_states(env, demo["states"])
        rewards = jax.jit(jax.vmap(lambda s: env._task.shaped_reward(s.data, *s.memory)))(states)
        np.testing.assert_allclose(np.asarray(rewards), expected, atol=1e-5, err_msg=name)


def test_steps_last_the_datasets_control_period(env):
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    states = _jit(env, "reset", vmap=True)(keys)
    actions = jax.vmap(env.action_space.sample)(keys) * 0.3
    step = _jit(env, "step", vmap=True)
    for _ in range(3):
        states = step(keys, states, actions)
    np.testing.assert_allclose(np.asarray(states.data.time), 3 * 0.05, rtol=1e-5)


@pytest.mark.parametrize("geom", ["box", "mesh"])
def test_contact_tests_count_contact_within_the_margin_and_slack_as_mujoco_does(geom):
    shape = 'type="box" size="0.02 0.02 0.02"' if geom == "box" else 'type="mesh" mesh="cube"'
    xml = f"""
    <mujoco>
      <asset><mesh name="cube" vertex="-.02 -.02 -.02  .02 -.02 -.02  -.02 .02 -.02  .02 .02 -.02
                                       -.02 -.02 .02  .02 -.02 .02  -.02 .02 .02  .02 .02 .02"/></asset>
      <worldbody>
        <body name="a"><freejoint/><geom name="a" {shape} margin="0.002"/></body>
        <body name="b" pos="0 0 0.1"><freejoint/><geom name="b" type="box" size="0.02 0.02 0.02"/></body>
      </worldbody>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    test = ContactTest(m, [m.geom("a").id], [m.geom("b").id])

    def touching(gap, slack=0.):
        d.qpos[7:10] = [0., 0., 0.04 + gap]
        mujoco.mj_forward(m, d)
        return bool(test(d, slack))

    assert touching(-0.001)
    assert touching(0.001)
    assert not touching(0.004)
    assert touching(0.004, slack=0.003)


@pytest.mark.parametrize("task", ["square_d1", "square_d2", "mug_cleanup_d1", "kitchen_d1", "hammer_cleanup_d1"])
def test_resets_place_every_geom_as_a_model_built_with_their_fixtures(task):
    env = RobomimicPOMDP(task, max_worlds=1)
    i = 7
    names = {f: [env.model.body(int(b)).name for b in index] for f, index in env._fixture_index.items()}
    fixtures = {f: np.asarray(x[i]) for f, x in zip(env._fixture_index, env._reset_fixtures)}
    built = RobomimicPOMDP(task, model_xml=generate._model_file(task, fixtures, names), max_worlds=1)
    reset = _jit(env, "reset_at")(i)
    expected = jax.jit(built.init_state)(env._reset_qpos[i], env._reset_qvel[i])
    np.testing.assert_allclose(np.asarray(reset.data.geom_xpos), np.asarray(expected.data.geom_xpos), atol=1e-5)
    np.testing.assert_allclose(np.asarray(reset.data.geom_xmat), np.asarray(expected.data.geom_xmat), atol=1e-5)
