"""
Tests for the MJX environment against robosuite, using the environment metadata of the test dataset.
Run with `uv run --extra mjx pytest tests/test_mjx.py`.
"""
from copy import deepcopy

import h5py
import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")

import jax
import mujoco
from mujoco import mjx

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.envs.env_base as EB
import robomimic.envs.env_mjx as EnvMJXModule
from robomimic.algo import RolloutPolicy
from robomimic.envs.wrappers import FrameStackWrapper

NUM_ENVS = 4
ROBOT_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]


@pytest.fixture(scope="module")
def env_meta():
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["object"] + ROBOT_KEYS, "rgb": []})
    return FileUtils.get_env_metadata_from_dataset(dataset_path=TestUtils.example_dataset_path())


@pytest.fixture(scope="module")
def cpu_env(env_meta):
    return EnvUtils.create_env_from_metadata(env_meta=deepcopy(env_meta), render=False, render_offscreen=False)


@pytest.fixture(scope="module")
def mjx_env(env_meta):
    meta = deepcopy(env_meta)
    meta["type"] = EB.EnvType.MJX_TYPE
    meta["env_kwargs"]["num_envs"] = NUM_ENVS
    return EnvUtils.create_env_from_metadata(env_meta=meta, render=False, render_offscreen=False)


def _cpu_episodes(cpu_env, actions):
    """Reset @cpu_env once per episode and play @actions of shape (T, num_envs, action_dim)."""
    states, obs = [], []
    for i in range(actions.shape[1]):
        traj = [cpu_env.reset()]
        states.append(cpu_env.get_state()["states"])
        for t in range(actions.shape[0]):
            traj.append(cpu_env.step(actions[t, i])[0])
        obs.append(traj)
    return np.stack(states), obs


def _free_space_actions(num_steps, seed=0):
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1., 1., size=(num_steps, NUM_ENVS, 7)) * np.array([.5, .5, .5, .2, .2, .2, 1.])
    actions[..., 2] = np.abs(actions[..., 2])
    actions[..., 6] = -1.
    return actions


class ScriptedPolicy(RolloutPolicy):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def start_episode(self):
        pass

    def __call__(self, ob, goal=None, batched_ob=False):
        batch_shape = (ob["robot0_eef_pos"].shape[0],) if batched_ob else ()
        return np.zeros(batch_shape + (self.action_dim,))


def test_env_type_resolves_to_mjx_class(mjx_env):
    assert EnvUtils.get_env_class(env_type=EB.EnvType.MJX_TYPE) is type(mjx_env)
    assert mjx_env.type == EB.EnvType.MJX_TYPE


def test_reset_to_observations_match_robosuite(cpu_env, mjx_env):
    states, cpu_obs = _cpu_episodes(cpu_env, np.zeros((0, NUM_ENVS, 7)))
    obs = mjx_env.reset_to({"states": states})
    for k in cpu_obs[0][0]:
        for i in range(NUM_ENVS):
            np.testing.assert_allclose(obs[k][i], cpu_obs[i][0][k], atol=1e-5, err_msg=k)


def test_robot_observations_track_robosuite_in_free_space(cpu_env, mjx_env):
    actions = _free_space_actions(num_steps=10)
    states, cpu_obs = _cpu_episodes(cpu_env, actions)
    mjx_env.reset_to({"states": states})
    for t in range(actions.shape[0]):
        obs = mjx_env.step(actions[t])[0]
        for k in ROBOT_KEYS:
            for i in range(NUM_ENVS):
                np.testing.assert_allclose(obs[k][i], cpu_obs[i][t + 1][k], atol=1e-3, err_msg="{} t={}".format(k, t))


def test_reset_to_own_state_restores_state_and_matches_robosuite(cpu_env, mjx_env):
    mjx_env.reset()
    mjx_env.step(_free_space_actions(num_steps=1)[0])
    state = mjx_env.get_state()
    obs = mjx_env.reset_to(state)
    np.testing.assert_allclose(mjx_env.get_state()["states"], state["states"], atol=1e-6)
    for i in range(NUM_ENVS):
        cpu_obs = cpu_env.reset_to({"states": state["states"][i]})
        for k in cpu_obs:
            np.testing.assert_allclose(obs[k][i], cpu_obs[k], atol=1e-5, err_msg=k)


def test_step_returns_batched_outputs(mjx_env):
    mjx_env.reset()
    obs, reward, done, info = mjx_env.step(np.zeros((NUM_ENVS, mjx_env.action_dimension)))
    assert all(v.shape[0] == NUM_ENVS for v in obs.values())
    assert reward.shape == (NUM_ENVS,)
    assert done.shape == (NUM_ENVS,)
    assert info["is_success"]["task"].shape == (NUM_ENVS,)
    assert info["diverged"].shape == (NUM_ENVS,)


def test_frame_stack_wrapper_stacks_after_batch_dimension(mjx_env):
    env = FrameStackWrapper(mjx_env, num_frames=3)
    obs = env.reset()
    assert obs["robot0_eef_pos"].shape == (NUM_ENVS, 3, 3)
    obs = env.step(np.zeros((NUM_ENVS, mjx_env.action_dimension)))[0]
    assert obs["robot0_eef_pos"].shape == (NUM_ENVS, 3, 3)
    assert obs["timesteps"].shape == (NUM_ENVS, 3, 1)
    assert obs["actions"].shape == (NUM_ENVS, 3, mjx_env.action_dimension)


def test_rollout_with_stats_runs_requested_number_of_episodes(mjx_env):
    horizon = 5
    logs, _ = TrainUtils.rollout_with_stats(
        policy=ScriptedPolicy(mjx_env.action_dimension),
        envs={"lift": mjx_env},
        horizon=horizon,
        num_episodes=NUM_ENVS + 1,
        verbose=True,
    )
    assert logs["lift"]["Horizon"] == horizon
    assert logs["lift"]["Success_Rate"] == 0.
    assert logs["lift"]["Diverged"] == 0.


def _contact_dists(xml, collision_functions=None):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cpu = sorted(d.contact[i].dist for i in range(d.ncon))
    with EnvMJXModule._collision_functions(collision_functions):
        dx = jax.jit(lambda mx, dx: mjx.forward(mx, dx))(mjx.put_model(m), mjx.put_data(m, d))
    dist = np.asarray(dx._impl.contact.dist)
    return cpu, sorted(dist[dist < 1.].tolist())


BOX_ON_TABLE = '<mujoco><worldbody><geom type="box" size=".4 .4 .02" pos="0 0 -.02"/>' \
    '<body pos="0 0 .019"><freejoint/><geom type="{}" size="{}"/></body></worldbody></mujoco>'


def test_cylinder_resting_on_box_penetrates_like_robosuite():
    cpu, mjx_dists = _contact_dists(BOX_ON_TABLE.format("cylinder", ".03 .02"))
    assert len(mjx_dists) > 0
    np.testing.assert_allclose(min(mjx_dists), min(cpu), atol=1e-5)


def test_cpu_contacts_reproduce_cpu_box_box_and_convex_contacts():
    funcs = EnvMJXModule._cpu_like_collision_functions()
    for geom, size in [("box", ".02 .02 .02"), ("cylinder", ".03 .02")]:
        cpu, mjx_dists = _contact_dists(BOX_ON_TABLE.format(geom, size), funcs)
        np.testing.assert_allclose(mjx_dists, cpu, atol=1e-5, err_msg=geom)


def _start_cpu_episode(cpu_env, xml, state):
    """Set @cpu_env to @state as if an episode started there: controllers refreshed, nullspace target at @state."""
    cpu_env.reset_to({"model": xml, "states": state})
    for robot in cpu_env.env.robots:
        for part, controller in robot.part_controllers.items():
            controller.new_update = True
            if part in robot.arms:
                controller.initial_joint = np.array(state[1:][controller.qpos_index])


def test_reset_to_per_env_demo_models_matches_robosuite(cpu_env, mjx_env):
    with h5py.File(TestUtils.example_dataset_path(), "r") as f:
        demos = sorted(f["data"], key=lambda k: int(k.split("_")[1]))[:NUM_ENVS]
        xmls = [f["data/{}".format(d)].attrs["model_file"] for d in demos]
        states = np.stack([f["data/{}/states".format(d)][0] for d in demos])
        actions = np.stack([f["data/{}/actions".format(d)][:5] for d in demos], axis=1)

    obs = mjx_env.reset_to({"states": states, "model": xmls})
    for i in range(NUM_ENVS):
        cpu_obs = cpu_env.reset_to({"model": xmls[i], "states": states[i]})
        for k in cpu_obs:
            np.testing.assert_allclose(obs[k][i], cpu_obs[k], atol=1e-5, err_msg="reset {} env {}".format(k, i))

    for t in range(actions.shape[0]):
        mjx_env.step(actions[t])
    mjx_states = mjx_env.get_state()["states"]
    for i in range(NUM_ENVS):
        _start_cpu_episode(cpu_env, xmls[i], states[i])
        for t in range(actions.shape[0]):
            cpu_env.step(actions[t, i])
        joints = 1 + np.asarray(cpu_env.env.robots[0]._ref_joint_pos_indexes)
        np.testing.assert_allclose(
            mjx_states[i][joints], cpu_env.get_state()["states"][joints], atol=1e-4, err_msg="env {}".format(i))
