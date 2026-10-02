"""
Tests of the MuJoCo Warp POMDPs against the robomimic demonstration datasets. The Lift test dataset (with
recorded camera images) is downloaded to tests/assets if missing; tests on the task datasets run when
$ROBOMIMIC_DATA/<task>/<ph|mh>/ holds demo_v15.hdf5 (raw) or low_dim_v15.hdf5 (with observations).
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax_pomdps as pomdps

from robomimic.data import hdf5, registry
from robomimic.env import TASK_OF_ENV_NAME, TASKS, RobomimicPOMDP, parked_joints

NUM_WORLDS = 4
CAMERAS = ["agentview", "robot0_eye_in_hand"]


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


def _task_dataset(task, filename="demo_v15.hdf5"):
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


def _episode_states(env, demo, starts):
    """States at the demo's steps `starts` with the controller state the demo had there."""
    grip = jnp.zeros((len(env.arms), 2), jnp.float32)
    grips = []
    for t in range(max(starts) + 1):
        grips.append(grip)
        action = jnp.asarray(demo["actions"][t], jnp.float32)
        grip = jnp.stack([arm.step_grip(g, action) for arm, g in zip(env.arms, grip)])
    q0 = tuple(jnp.asarray(demo["states"][0][1:][arm.qpos], jnp.float32) for arm in env.arms)
    states = jax.jit(jax.vmap(env.state_from_flat))(demo["states"][starts])
    return states._replace(grip=jnp.stack([grips[t] for t in starts]), q0=tuple(jnp.broadcast_to(q, (len(starts),) + q.shape) for q in q0))


def _live_qpos(m, flat):
    """qpos indices of everything but the objects robosuite parks far outside the workspace at state `flat`."""
    parked = [np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + 7) for j in parked_joints(m, flat[1:1 + m.nq])]
    return np.setdiff1d(np.arange(m.nq), np.concatenate(parked) if parked else [])


@pytest.fixture(scope="module", params=list(TASKS))
def env(request):
    return pomdps.make("robomimic.env:robomimic/" + request.param, max_worlds=NUM_WORLDS)


def test_tasks_are_registered_pomdps(env):
    assert isinstance(env, pomdps.POMDP)
    assert env.action_space.shape == (7 * len(env.arms),)


def test_unknown_tasks_are_rejected():
    with pytest.raises(ValueError):
        RobomimicPOMDP("lift_real")


def test_batched_reset_step_observe_under_jit_and_vmap(env):
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.vmap(env.action_space.sample)(keys) * 0.3
    next_states = jax.jit(jax.vmap(env.step))(keys, states, actions)
    obs = jax.jit(jax.vmap(env.observe))(keys, next_states, actions)
    rewards = jax.jit(jax.vmap(env.reward))(keys, states, actions, next_states)
    dones = jax.jit(jax.vmap(env.done))(next_states)
    assert rewards.shape == (NUM_WORLDS,)
    assert dones.shape == (NUM_WORLDS,)
    assert not bool(jnp.any(jax.vmap(env.overflowed)(next_states)))
    for k, space in env.observation_space.spaces.items():
        assert obs[k].shape == (NUM_WORLDS,) + space.shape, k
        assert bool(jnp.all(jnp.isfinite(obs[k]))), k


def test_flat_state_round_trips(env):
    state = jax.jit(env.reset)(jax.random.key(1))
    flat = env.flat_state(state)
    np.testing.assert_allclose(env.flat_state(jax.jit(env.state_from_flat)(flat)), flat, atol=1e-6)


@pytest.mark.parametrize("task", list(TASKS))
def test_observations_match_dataset(task):
    """Observations at recorded states equal the recorded ones; robosuite recorded each state's observation
    right after the previous state's, which object poses relative to the gripper are cached from."""
    demo = _demo(_task_dataset(task, "low_dim_v15.hdf5"), "demo_0")
    steps = np.linspace(1, len(demo["states"]) - 1, NUM_WORLDS).astype(int)
    env = RobomimicPOMDP(task, model_xml=demo["model"], max_worlds=NUM_WORLDS)
    init = jax.jit(jax.vmap(env.state_from_flat))
    previous = init(demo["states"][steps - 1])
    states = init(demo["states"][steps])._replace(cache=previous.cache)
    obs = jax.jit(jax.vmap(env.observe))(None, states, None)
    assert set(obs) == set(demo["obs"])
    for k in obs:
        np.testing.assert_allclose(np.asarray(obs[k]), demo["obs"][k][steps], atol=1e-5, err_msg=k)


def test_camera_images_resemble_dataset_images(lift_test_dataset):
    demo = _demo(lift_test_dataset, "demo_0")
    height, width = demo["obs"]["agentview_image"].shape[1:3]
    env = RobomimicPOMDP("lift", model_xml=demo["model"], camera_names=CAMERAS, camera_height=height,
                         camera_width=width, camera_depths=True, max_worlds=NUM_WORLDS)
    steps = np.linspace(0, len(demo["states"]) - 1, NUM_WORLDS).astype(int)
    states = jax.jit(jax.vmap(env.state_from_flat))(demo["states"][steps])
    obs = jax.jit(jax.vmap(env.observe))(None, states, None)
    for camera in CAMERAS:
        image, expected = np.asarray(obs[camera + "_image"]), demo["obs"][camera + "_image"][steps]
        assert image.dtype == np.uint8 and image.shape == expected.shape
        assert np.abs(image.astype(float) - expected).mean() < 8., camera
        depth, expected = np.asarray(obs[camera + "_depth"]), demo["obs"][camera + "_depth"][steps]
        assert np.median(np.abs(depth - expected)) < 0.01, camera


def test_camera_observations_match_their_spaces_unbatched_and_mapped():
    env = RobomimicPOMDP("lift", camera_names=CAMERAS, camera_height=48, camera_width=64, camera_depths=True,
                         max_worlds=NUM_WORLDS)
    space = env.observation_space.spaces
    assert isinstance(space["agentview_image"], pomdps.spaces.Image)
    assert space["agentview_image"].shape == (48, 64, 3)
    assert space["robot0_eye_in_hand_depth"].shape == (48, 64, 1)
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    mapped = jax.jit(jax.vmap(env.observe))(keys, jax.jit(jax.vmap(env.reset))(keys), None)
    single = jax.jit(env.observe)(keys[1], jax.jit(env.reset)(keys[1]), None)
    for k in ["agentview_image", "robot0_eye_in_hand_image", "agentview_depth", "robot0_eye_in_hand_depth"]:
        assert mapped[k].shape == (NUM_WORLDS,) + space[k].shape, k
        np.testing.assert_array_equal(np.asarray(single[k]), np.asarray(mapped[k][1]), err_msg=k)
    assert int(jnp.ptp(mapped["agentview_image"])) > 0


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
    env = RobomimicPOMDP(task, model_xml=demo["model"], max_worlds=2)
    states = jax.jit(jax.vmap(env.state_from_flat))(demo["states"][[0, -1]])
    first, last = np.asarray(jax.vmap(env.success)(states))
    assert not first and last


@pytest.mark.parametrize("task", list(TASKS))
def test_one_step_transitions_match_dataset(task):
    """Each demo transition, simulated from its recorded state and controller state, lands on the next recorded state
    in the coordinates that take part in the task."""
    demo = _demo(_task_dataset(task), "demo_0")
    starts = np.arange(len(demo["actions"]) - 1)
    env = RobomimicPOMDP(task, model_xml=demo["model"], max_worlds=len(starts))
    states = _episode_states(env, demo, starts)
    next_states = jax.jit(jax.vmap(env.step))(None, states, jnp.asarray(demo["actions"][starts], jnp.float32))
    assert not bool(jnp.any(jax.vmap(env.overflowed)(next_states)))
    live = _live_qpos(env.model, demo["states"][0])
    error = np.abs(np.asarray(next_states.data.qpos)[:, live] - demo["states"][starts + 1][:, 1:][:, live])
    assert np.median(error.max(axis=1)) < 1e-3
    assert np.percentile(error.max(axis=1), 95) < 1e-2




def test_steps_last_the_datasets_control_period(env):
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.vmap(env.action_space.sample)(keys) * 0.3
    step = jax.jit(jax.vmap(env.step))
    for _ in range(3):
        states = step(keys, states, actions)
    np.testing.assert_allclose(np.asarray(states.data.time), 3 * 0.05, rtol=1e-5)
