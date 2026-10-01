"""
Robomimic's robosuite tasks as jax_mdps POMDPs simulated by MuJoCo Warp through MJX, registered as
"robomimic/lift", "robomimic/can", "robomimic/square", "robomimic/transport", and "robomimic/tool_hang":

    import jax_mdps as mdps
    env = mdps.make("robomimic.pomdp:robomimic/lift")

Each task's model, controller parameters, task geometry, and initial states come from `data/`, exported
from robosuite v1.5. Physics runs in MuJoCo Warp; the arm and gripper controllers, observations, rewards, and
success run in JAX, and camera images come from the MuJoCo Warp ray tracer. All methods are pure functions of
pytrees, safe under `jax.jit` and `jax.vmap`.
"""
import functools
import json
import os
import posixpath
import re
import zipfile
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx.third_party.mujoco_warp import OverflowType

import jax_mdps as mdps
import warp as wp

from robomimic.control import Arm
from robomimic.render import CameraRenderer
from robomimic.tasks import TASKS

_DATA = Path(__file__).parent / "data"

# free objects farther than this from the world origin at reset are parked by the task and take no part in it
_PARKED_DISTANCE = 5.
# joint damping that holds parked objects in place
_PARKED_DAMPING = 1e4

# share of the contact buffer usable by contacts that need EPA, at least twice the measured peaks
_CCD_SHARE = 0.25
# overflows of fixed-size MuJoCo Warp buffers, which drop contacts or constraints
_BUFFER_OVERFLOWS = int(
    OverflowType.NEFC | OverflowType.NJMAX_NNZ | OverflowType.BROADPHASE
    | OverflowType.NARROWPHASE | OverflowType.CCD | OverflowType.EPA_HORIZON
)
# default (contacts, constraint rows) allocated per world, at least twice the measured peaks
_BUFFER_SIZES = {
    "lift": (32, 256),
    "can": (32, 256),
    "square": (64, 512),
    "tool_hang": (128, 512),
    "transport": (192, 1024),
}


def _keep_external_streams_registered():
    """
    Keep CUDA streams that Warp does not own (JAX's) registered with Warp for the life of the process.
    Warp wraps JAX's stream anew on every FFI call, and freeing any one wrapper must not unregister the
    stream while another wrapper is using it.
    """
    release = wp.Stream.__del__

    def __del__(self):
        if getattr(self, "owner", False):
            release(self)

    wp.Stream.__del__ = __del__


_keep_external_streams_registered()


@functools.cache
def _archive():
    """Contents of data/models.zip: the task models and the meshes and textures they use."""
    with zipfile.ZipFile(_DATA / "models.zip") as z:
        return {name: z.read(name) for name in z.namelist()}


@functools.cache
def _specs():
    return json.loads((_DATA / "tasks.json").read_text())


# task of each robosuite environment name in the robomimic datasets
TASK_OF_ENV_NAME = {s["env_name"]: t for t, s in _specs().items()}


def load_model(task: str | None = None, model_xml: str | None = None) -> mujoco.MjModel:
    """
    MuJoCo model of `task`, or of a robosuite model xml such as a dataset demo's "model_file", whose asset
    paths are resolved to the bundled meshes and textures.
    """
    if model_xml is None:
        model_xml = _archive()[f"models/{task}.xml"].decode()
    model_xml = re.sub(
        r'file="(?:[^"]*/)?robosuite/([^"]*)"', lambda match: f'file="{posixpath.normpath(match[1])}"', model_xml
    )
    return mujoco.MjModel.from_xml_string(model_xml, _archive())


def parked_joints(m, qpos) -> np.ndarray:
    """Free joints of `m` whose bodies are parked outside the workspace at joint positions `qpos`."""
    free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    return np.array(
        [j for j in free if np.linalg.norm(qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3]) > _PARKED_DISTANCE], dtype=int
    )


def _park(m, qpos):
    """Copy of `m` in which objects parked outside the workspace at `qpos` neither collide nor move."""
    m = deepcopy(m)
    joints = parked_joints(m, qpos)
    for j in joints:
        m.dof_damping[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = _PARKED_DAMPING
    parked = np.isin(m.body_rootid[m.geom_bodyid], m.jnt_bodyid[joints])
    m.geom_contype[parked] = 0
    m.geom_conaffinity[parked] = 0
    return m


def _split_flat(flat, nq, nv):
    """(time, qpos, qvel) of robosuite's flattened simulator state `flat`."""
    return flat[0], flat[1:1 + nq], flat[1 + nq:1 + nq + nv]


def _space(key, struct):
    """Space of observation `key`, of shape and dtype `struct`."""
    if struct.dtype == jnp.uint8:
        return mdps.spaces.Image(*struct.shape)
    if key.endswith("_depth"):
        return mdps.spaces.Box(0., np.inf, struct.shape)
    return mdps.spaces.Box(-np.inf, np.inf, struct.shape)


class State(NamedTuple):
    data: mjx.Data
    grip: jax.Array
    q0: tuple[jax.Array, ...]
    cache: Any


class RobomimicPOMDP:
    """A robomimic task whose physics runs in MuJoCo Warp."""
    def __init__(
        self,
        task: str,
        model_xml: str | None = None,
        camera_names: Sequence[str] = (),
        camera_height: int = 84,
        camera_width: int = 84,
        camera_depths: bool = False,
        max_worlds: int = 4096,
        contacts_per_world: int | None = None,
        constraints_per_world: int | None = None,
        terminate_on_success: bool = True,
        graph_mode: Literal["jax", "warp", "none"] = "warp",
    ):
        """
        Simulates `model_xml`, a robosuite model xml such as a dataset demo's "model_file", instead of the
        task's own model when given. Cameras are observed under "<camera>_image" and, with `camera_depths`,
        "<camera>_depth". At most `max_worlds` worlds are simulated together; with cameras, `observe` is
        unbatched or mapped over exactly `max_worlds` states. `contacts_per_world` is an average over the worlds.
        """
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}, not {task!r}")
        spec = _specs()[task]
        m = load_model(task, model_xml)
        self.arms = tuple(Arm(a, m) for a in spec["arms"])
        self.model = m
        self._task = TASKS[task](spec["task"], m, self.arms)
        self._n_substeps = spec["n_substeps"]
        self._terminate_on_success = terminate_on_success

        with np.load(_DATA / "resets.npz") as resets:
            self._reset_qpos = jnp.asarray(resets[f"{task}_qpos"], jnp.float32)
            self._reset_qvel = jnp.asarray(resets[f"{task}_qvel"], jnp.float32)

        self._m = _park(m, np.asarray(self._reset_qpos[0]))
        graph_mode = getattr(wp.JaxCallableGraphMode, graph_mode.upper())
        self._mx = mjx.put_model(self._m, impl="warp", graph_mode=graph_mode)
        self._renderer = None
        if camera_names:
            self._renderer = CameraRenderer(
                self._m, camera_names=camera_names, height=camera_height, width=camera_width, depth=camera_depths,
                nworld=max_worlds, graph_mode=graph_mode,
            )
        default_contacts, default_constraints = _BUFFER_SIZES[task]
        self._naconmax = int(max_worlds * (contacts_per_world or default_contacts))
        self._njmax = int(constraints_per_world or default_constraints)

        self.action_space = mdps.spaces.Box(-1., 1., (spec["action_dim"],))
        obs_shapes = jax.eval_shape(lambda s: self.observe(None, s, None), jax.eval_shape(self.reset, jax.random.key(0)))
        self.observation_space = mdps.spaces.Dict({k: _space(k, v) for k, v in obs_shapes.items()})

    def init_state(self, qpos, qvel, time=0.):
        """The state at joint positions `qpos`, velocities `qvel`, and `time`, as at the start of an episode."""
        d = mjx.make_data(
            self._m, impl="warp", naconmax=self._naconmax, naccdmax=int(self._naconmax * _CCD_SHARE), njmax=self._njmax
        )
        d = d.replace(
            qpos=jnp.asarray(qpos, jnp.float32), qvel=jnp.asarray(qvel, jnp.float32), time=jnp.asarray(time, d.time.dtype)
        )
        d = mjx.forward(self._mx, d)
        grip = jnp.zeros((len(self.arms), 2), jnp.float32)
        q0 = tuple(d.qpos[arm.qpos] for arm in self.arms)
        return State(d, grip, q0, self._task.cache(d))

    def reset(self, key):
        i = jax.random.randint(key, (), 0, self._reset_qpos.shape[0])
        return self.init_state(self._reset_qpos[i], self._reset_qvel[i])

    def step(self, key, state, action):
        m = self._mx
        action = jnp.asarray(action, jnp.float32)
        d = mjx.forward(m, state.data)
        grip = jnp.stack([arm.step_grip(g, action) for arm, g in zip(self.arms, state.grip)])
        controls = [(arm.goal(d, action), arm.grip_ctrl(g), q0) for arm, g, q0 in zip(self.arms, grip, state.q0)]

        def substep(d, _):
            d = mjx.forward(m, d)
            ctrl = d.ctrl
            for arm, ((goal_pos, goal_ori), grip_ctrl, q0) in zip(self.arms, controls):
                ctrl = ctrl.at[arm.actuators].set(arm.torques(m, d, goal_pos, goal_ori, q0))
                ctrl = ctrl.at[arm.grip_actuators].set(grip_ctrl)
            return mjx.step(m, d.replace(ctrl=ctrl)), None

        d, _ = jax.lax.scan(substep, d, None, length=self._n_substeps)
        return state._replace(data=d, grip=grip, cache=self._task.cache(state.data))

    def observe(self, key, next_state, action):
        d = next_state.data
        obs = {"object": self._task.object_obs(d, next_state.cache)}
        for arm in self.arms:
            obs |= arm.observe(d)
        if self._renderer is not None:
            obs |= self._renderer(d)
        return obs

    def reward(self, key, state, action, next_state):
        return self._task.reward(next_state.data)

    def done(self, state):
        return self._terminate_on_success & self.success(state)

    def success(self, state):
        """Whether the task has been accomplished in `state`."""
        return self._task.success(state.data)

    def overflowed(self, state):
        """Whether contact or constraint buffers overflowed while simulating `state`, dropping contacts or constraints."""
        return (state.data._impl.overflow & _BUFFER_OVERFLOWS) != 0

    def flat_state(self, state):
        """robosuite's flattened simulator state [time, qpos, qvel] of `state`."""
        d = state.data
        return np.concatenate([np.asarray(d.time).reshape(1), np.asarray(d.qpos), np.asarray(d.qvel)])

    def state_from_flat(self, flat):
        """The state at robosuite's flattened simulator state `flat`, as at the start of an episode."""
        time, qpos, qvel = _split_flat(flat, self._m.nq, self._m.nv)
        return self.init_state(qpos, qvel, time)

    def render_flat(self, flat, camera_name="agentview", height=256, width=256):
        """RGB frame of robosuite's flattened simulator state `flat`, rendered by MuJoCo's OpenGL renderer on the CPU."""
        m = self.model
        d = mujoco.MjData(m)
        d.time, d.qpos[:], d.qvel[:] = _split_flat(np.asarray(flat, dtype=np.float64), m.nq, m.nv)
        mujoco.mj_forward(m, d)
        option = mujoco.MjvOption()
        option.geomgroup[0] = 0
        option.sitegroup[:] = 0
        with mujoco.Renderer(m, height=height, width=width) as renderer:
            renderer.update_scene(d, camera=camera_name, scene_option=option)
            return renderer.render()


for _task in TASKS:
    mdps.register(f"robomimic/{_task}", functools.partial(RobomimicPOMDP, _task))
