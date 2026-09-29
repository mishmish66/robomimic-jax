"""
This file contains an MJX (MuJoCo XLA) environment wrapper that simulates a batch of
robosuite task instances in parallel on an accelerator. Physics and the arm / gripper
controllers run in JAX. A regular robosuite environment is kept on the CPU as the source of
the scene model, controller parameters, and reset distribution; it also evaluates
observations, rewards, and success for each batch member from that member's simulated
state, and renders frames.

All observations, rewards, and success flags carry a leading batch dimension of size
@num_envs.
"""
import os
import json
from copy import deepcopy

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp

# contact dynamics of light objects need double precision unless the user chooses otherwise
if "JAX_ENABLE_X64" not in os.environ:
    jax.config.update("jax_enable_x64", True)
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_convex as mjx_collision_convex
from mujoco.mjx._src import collision_driver as mjx_collision_driver
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import mesh as mjx_mesh
from mujoco.mjx._src import solver as mjx_solver
from mujoco.mjx._src.collision_types import ConvexInfo

import robomimic.envs.env_base as EB
from robomimic.envs.env_robosuite import EnvRobosuite


# collision bitmasks: arena <-> {object, gripper}, object <-> {object, gripper}
_ARENA_BIT, _OBJECT_BIT, _GRIPPER_BIT = 1, 2, 4
_COLLISION_MASKS = {
    "arena": (_ARENA_BIT, 0),
    "object": (_OBJECT_BIT, _ARENA_BIT | _OBJECT_BIT | _GRIPPER_BIT),
    "gripper": (_GRIPPER_BIT, _ARENA_BIT),
}

# free objects farther than this from the world origin at reset are parked by the task and take no part in it
_PARKED_DISTANCE = 5.
# joint damping that holds parked objects in place
_PARKED_DAMPING = 1e4

# default MJX broadphase limits: closest geom pairs kept per collision function, deepest contacts kept per condim
_DEFAULT_MAX_GEOM_PAIRS = 128
_DEFAULT_MAX_CONTACT_POINTS = 128

# number of sides of the convex prism that stands in for a cylinder in cylinder-box and cylinder-mesh collisions
_CYLINDER_PRISM_SIDES = 16

# model fields that only affect rendering
_VISUAL_FIELD_PREFIXES = (
    "site_rgba", "geom_rgba", "tendon_rgba", "flex_rgba", "mat_", "tex_", "light_", "cam_", "skin_", "mesh_polynormal",
)


def _unit_prism(sides):
    """
    Convex prism approximating the cylinder of radius 1 and half-height 1 along z, as
    (vert, face, face_normal, edge, edge_face_normal) in the layout of MJX convex geoms.
    """
    angles = np.arange(sides) * 2. * np.pi / sides
    radius = 2. / (1. + np.cos(np.pi / sides))
    ring = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vert = np.concatenate([
        np.concatenate([ring, np.full((sides, 1), z)], axis=1) for z in (-1., 1.)
    ])
    k = np.arange(sides)
    walls = np.stack([k, (k + 1) % sides, sides + (k + 1) % sides, sides + k], axis=1)
    walls = np.pad(walls, ((0, 0), (0, sides - 4)), mode="edge")
    face = np.concatenate([k[::-1][None], (sides + k)[None], walls])
    face_normal = mjx_mesh._get_face_norm(vert, face)
    edge, edge_face_normal = mjx_mesh._get_edge_normals(face, face_normal)
    return vert, vert[face], face_normal, edge, edge_face_normal


_UNIT_PRISM = _unit_prism(_CYLINDER_PRISM_SIDES)


@mjx_collision_convex.collider(ncon=4)
def _cylinder_convex(cylinder, convex):
    """Contacts between a cylinder, treated as a convex prism, and a box or convex mesh."""
    vert, face, face_normal, edge, edge_face_normal = (jnp.asarray(x) for x in _UNIT_PRISM)
    scale = jnp.array([cylinder.size[0], cylinder.size[0], cylinder.size[1]])
    prism = ConvexInfo(
        cylinder.pos, cylinder.mat, cylinder.size, vert * scale, face * scale, face_normal, edge, edge_face_normal,
    )
    dist, pos, normal = mjx_collision_convex._convex_convex(prism, convex)
    return dist, pos, jax.vmap(mjx_math.make_frame)(normal)


for _other in (mjx.GeomType.BOX, mjx.GeomType.MESH):
    mjx_collision_driver._COLLISION_FUNC.setdefault((mjx.GeomType.CYLINDER, _other), _cylinder_convex)


def _deepest_contact_only(collide):
    """Collision function that keeps only the deepest of @collide's contacts per geom pair."""
    def wrapped(m, d, key, geom):
        dist, pos, frame = collide(m, d, key, geom)
        dist = dist.reshape(-1, collide.ncon)
        keep = jnp.arange(collide.ncon)[None] == jnp.argmin(dist, axis=1)[:, None]
        dist = jnp.where(keep, dist, jnp.finfo(dist.dtype).max).reshape(-1)
        return dist, pos, frame
    wrapped.ncon = collide.ncon
    return wrapped


def _half_depth(collide):
    """Collision function that reports half of @collide's penetration depth."""
    def wrapped(m, d, key, geom):
        dist, pos, frame = collide(m, d, key, geom)
        return 0.5 * dist, pos, frame
    wrapped.ncon = collide.ncon
    return wrapped


def _cpu_like_collision_functions():
    """
    MJX collision functions that reproduce CPU MuJoCo's contacts without multiccd: box-box contacts
    report half the penetration depth, and convex pairs involving a mesh or cylinder yield one contact.
    """
    G = mjx.GeomType
    funcs = dict(mjx_collision_driver._COLLISION_FUNC)
    funcs[(G.BOX, G.BOX)] = _half_depth(funcs[(G.BOX, G.BOX)])
    for pair in [(G.BOX, G.MESH), (G.MESH, G.MESH), (G.CYLINDER, G.BOX), (G.CYLINDER, G.MESH)]:
        funcs[pair] = _deepest_contact_only(funcs[pair])
    return funcs


class _collision_functions:
    """Context manager that installs the MJX collision function table @funcs while active."""
    def __init__(self, funcs):
        self._funcs = funcs

    def __enter__(self):
        self._saved = dict(mjx_collision_driver._COLLISION_FUNC)
        if self._funcs is not None:
            mjx_collision_driver._COLLISION_FUNC.update(self._funcs)

    def __exit__(self, *args):
        mjx_collision_driver._COLLISION_FUNC.clear()
        mjx_collision_driver._COLLISION_FUNC.update(self._saved)


def _skew(v):
    return jnp.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.],
    ])


def _axisangle2mat(v):
    angle = jnp.linalg.norm(v)
    nonzero = angle > 0.
    axis = v / jnp.where(nonzero, angle, 1.)
    K = _skew(axis)
    R = jnp.eye(3) + jnp.sin(angle) * K + (1. - jnp.cos(angle)) * (K @ K)
    return jnp.where(nonzero, R, jnp.eye(3))


def _orientation_error(desired, current):
    return 0.5 * (
        jnp.cross(current[:, 0], desired[:, 0])
        + jnp.cross(current[:, 1], desired[:, 1])
        + jnp.cross(current[:, 2], desired[:, 2])
    )


def _float_fields(m):
    """Names of the floating point array fields of a mujoco.MjModel."""
    names = []
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            value = getattr(m, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.size > 0:
            names.append(name)
    return names


class _Arm:
    """Controller parameters and model indices for one OSC_POSE arm and its gripper."""
    def __init__(self, robot, arm, action_offset, m):
        osc = robot.part_controllers[arm]
        assert type(osc).__name__ == "OperationalSpaceController", "MJX mode requires OSC arm controllers"
        assert osc.use_ori and osc.input_type == "delta" and osc.impedance_mode == "fixed"
        assert osc.interpolator_pos is None and osc.interpolator_ori is None
        assert osc.position_limits is None and not np.array(osc.orientation_limits).any()

        grip_name = "{}_gripper".format(arm)
        grip_ctrl = robot.part_controllers[grip_name]
        assert type(grip_ctrl).__name__ == "SimpleGripController"
        gripper = robot.gripper[arm]
        assert type(gripper).__name__ == "PandaGripper", "MJX mode supports the Panda gripper"

        split = robot.composite_controller._action_split_indexes
        self.arm_action = slice(action_offset + split[arm][0], action_offset + split[arm][1])
        self.grip_action = slice(action_offset + split[grip_name][0], action_offset + split[grip_name][1])

        def affine(ctrl):
            in_min, in_max = np.asarray(ctrl.input_min, float), np.asarray(ctrl.input_max, float)
            out_min, out_max = np.asarray(ctrl.output_min, float), np.asarray(ctrl.output_max, float)
            scale = np.abs(out_max - out_min) / np.abs(in_max - in_min)
            return (jnp.asarray(in_min), jnp.asarray(in_max), jnp.asarray((in_max + in_min) / 2.),
                    jnp.asarray(scale), jnp.asarray((out_max + out_min) / 2.))

        self.arm_scale = affine(osc)
        self.grip_scale = affine(grip_ctrl)

        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, osc.ref_name)
        self.site_body = int(m.site_bodyid[self.site])
        self.base_frame = osc.input_ref_frame == "base"
        self.base_site = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "{}{}_center".format(osc.naming_prefix, osc.part_name)
        )
        self.dofs = jnp.asarray(np.asarray(osc.qvel_index, dtype=int))
        self.qpos = np.asarray(osc.qpos_index, dtype=int)
        self.kp = jnp.asarray(osc.kp, dtype=float)
        self.kd = jnp.asarray(osc.kd, dtype=float)
        self.uncouple = bool(osc.uncoupling)

        acts = robot._ref_actuators_indexes_dict
        self.arm_act = jnp.asarray(np.asarray(acts[arm], dtype=int))
        self.grip_act = jnp.asarray(np.asarray(acts[grip_name], dtype=int))
        ctrlrange = m.actuator_ctrlrange
        self.arm_ctrl_lo = jnp.asarray(ctrlrange[acts[arm], 0])
        self.arm_ctrl_hi = jnp.asarray(ctrlrange[acts[arm], 1])
        self.grip_ctrl_lo = jnp.asarray(ctrlrange[acts[grip_name], 0])
        self.grip_ctrl_hi = jnp.asarray(ctrlrange[acts[grip_name], 1])
        self.grip_speed = float(gripper.speed)
        self.grip_sign = jnp.asarray(np.array([-1., 1.]))

    @staticmethod
    def _scale(action, params):
        in_min, in_max, in_mid, scale, out_mid = params
        return (jnp.clip(action, in_min, in_max) - in_mid) * scale + out_mid

    def goal(self, d, action):
        """World-frame OSC goal pose for this arm's slice of @action."""
        delta = self._scale(action[self.arm_action], self.arm_scale)
        if self.base_frame:
            origin_ori = d.site_xmat[self.base_site].reshape(3, 3)
        else:
            origin_ori = jnp.eye(3)
        ref_pos = d.site_xpos[self.site]
        ref_ori = d.site_xmat[self.site].reshape(3, 3)
        goal_pos = ref_pos + origin_ori @ delta[:3]
        goal_ori = origin_ori @ _axisangle2mat(delta[3:6]) @ origin_ori.T @ ref_ori
        return goal_pos, goal_ori

    def torques(self, m, d, goal_pos, goal_ori, q0):
        dofs = self.dofs
        ref_pos = d.site_xpos[self.site]
        ref_ori = d.site_xmat[self.site].reshape(3, 3)
        jacp, jacr = mjx.jac(m, d, ref_pos, self.site_body)
        vel_pos = jacp.T @ d.qvel
        vel_ori = jacr.T @ d.qvel
        J_pos = jacp[dofs].T
        J_ori = jacr[dofs].T
        J_full = jnp.concatenate([J_pos, J_ori], axis=0)
        mass = mjx.full_m(m, d)[dofs][:, dofs]
        joint_pos = d.qpos[self.qpos]
        joint_vel = d.qvel[dofs]

        force = self.kp[:3] * (goal_pos - ref_pos) - self.kd[:3] * vel_pos
        torque = self.kp[3:] * _orientation_error(goal_ori, ref_ori) - self.kd[3:] * vel_ori

        mass_inv = jnp.linalg.inv(mass)
        lambda_full = jnp.linalg.pinv(J_full @ mass_inv @ J_full.T)
        if self.uncouple:
            lambda_pos = jnp.linalg.pinv(J_pos @ mass_inv @ J_pos.T)
            lambda_ori = jnp.linalg.pinv(J_ori @ mass_inv @ J_ori.T)
            wrench = jnp.concatenate([lambda_pos @ force, lambda_ori @ torque])
        else:
            wrench = lambda_full @ jnp.concatenate([force, torque])
        jbar = mass_inv @ J_full.T @ lambda_full
        nullspace = jnp.eye(J_full.shape[1]) - jbar @ J_full

        joint_kp = 10.
        pose_torques = mass @ (joint_kp * (q0 - joint_pos) - 2. * jnp.sqrt(joint_kp) * joint_vel)
        torques = J_full.T @ wrench + d.qfrc_bias[dofs] + nullspace.T @ pose_torques
        return jnp.clip(torques, self.arm_ctrl_lo, self.arm_ctrl_hi)

    def step_grip(self, grip, action):
        """Advance the Panda gripper's binary open / close state by one policy step."""
        return jnp.clip(grip + self.grip_sign * self.grip_speed * jnp.sign(action[self.grip_action]), -1., 1.)

    def grip_ctrl(self, grip):
        goal = self._scale(grip, self.grip_scale)
        bias = 0.5 * (self.grip_ctrl_hi + self.grip_ctrl_lo)
        weight = 0.5 * (self.grip_ctrl_hi - self.grip_ctrl_lo)
        return jnp.clip(bias + weight * goal, self.grip_ctrl_lo, self.grip_ctrl_hi)


class EnvMJX(EB.EnvBase):
    """Batched MJX simulation of a robosuite task (fixed-base Panda arms, OSC_POSE + GRIP)."""
    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
        lang=None,
        num_envs=16,
        mjx_config=None,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of robosuite environment

            render (bool): must be False; on-screen rendering is not supported

            render_offscreen (bool): if True, @render can produce frames of one batch member

            use_image_obs (bool): must be False; image observations are not supported

            use_depth_obs (bool): must be False; depth observations are not supported

            lang: language description for the environment

            num_envs (int): number of environments simulated in parallel

            mjx_config (dict): optional overrides for the MJX model options. Supported keys:
                "iterations" (int), "ls_iterations" (int), "tolerance" (float), "cone" ("pyramidal" or "elliptic"),
                "collide_arm_with_arena" (bool),
                "cpu_contacts" (bool, default True) to reproduce CPU MuJoCo's box-box depths and single convex contacts,
                "max_geom_pairs" (int) closest geom pairs checked per collision function, and
                "max_contact_points" (int) deepest contacts kept per contact dimension (-1 for no limit).

            kwargs: robosuite environment kwargs (same as @EnvRobosuite)
        """
        assert not render, "MJX mode does not support on-screen rendering"
        assert not (use_image_obs or use_depth_obs), "MJX mode does not support image observations"

        self._env_name = env_name
        self._num_envs = int(num_envs)
        self._mjx_config = dict(mjx_config or {})
        self._init_kwargs = deepcopy(kwargs)

        cpu_kwargs = deepcopy(kwargs)
        cpu_kwargs["hard_reset"] = False
        self._cpu = EnvRobosuite(
            env_name=env_name,
            render=False,
            render_offscreen=render_offscreen,
            use_image_obs=False,
            use_depth_obs=False,
            lang=lang,
            **cpu_kwargs,
        )
        rs = self._cpu.env
        self._n_substeps = int(round(rs.control_timestep / rs.model_timestep))
        self._setup_arms()
        self._parked_bodies = self._find_parked_bodies()

        self._device = jax.devices()[0]
        self._collision_funcs = _cpu_like_collision_functions() if self._mjx_config.get("cpu_contacts", True) else None
        self._model_cache = {}
        self._jit_cache = {}
        self._cpu_defaults = {}
        self._xmls = None
        self._cpu_xml = rs.model.get_xml()
        self._load_models([self._cpu_xml] * self._num_envs)

        self._d = None
        self._grip = None
        self._q0 = None
        self._obs = None
        self._reward = None
        self._success = None

    # ---------------------------------------------------------------------------------
    # models and controller parameters
    # ---------------------------------------------------------------------------------

    def _setup_arms(self):
        rs = self._cpu.env
        m = rs.sim.model._model
        self._arms = []
        offset = 0
        for robot in rs.robots:
            assert type(robot.composite_controller).__name__ == "CompositeController", \
                "MJX mode requires the BASIC composite controller"
            assert set(robot.part_controllers) == set(robot.arms) | {"{}_gripper".format(a) for a in robot.arms}, \
                "MJX mode supports arm and gripper parts only"
            for arm in robot.arms:
                self._arms.append(_Arm(robot, arm, offset, m))
            offset += robot.action_dim
        self._robot_roots = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.root_body) for robot in rs.robots
        }
        self._gripper_prefixes = tuple(g.naming_prefix for robot in rs.robots for g in robot.gripper.values())

    def _find_parked_bodies(self):
        """Names of the free bodies the task has placed outside the workspace in the CPU simulation."""
        m, d = self._cpu.env.sim.model._model, self._cpu.env.sim.data._data
        names = set()
        for j in range(m.njnt):
            adr = m.jnt_qposadr[j]
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and np.linalg.norm(d.qpos[adr:adr + 3]) > _PARKED_DISTANCE:
                names.add(m.body(m.jnt_bodyid[j]).name)
        return names

    def _collision_group(self, m, body):
        """Collision group of @body: "arena", "object", "gripper", or None for no collisions."""
        root = m.body_rootid[body]
        if m.body(root).name in self._parked_bodies:
            return None
        if root in self._robot_roots:
            if m.body(body).name.startswith(self._gripper_prefixes):
                return "gripper"
            return "gripper" if self._mjx_config.get("collide_arm_with_arena", False) else None
        if m.body_weldid[body] == 0:
            return "arena"
        return "object"

    def _compile_mjx_model(self, xml):
        """
        Original MuJoCo model compiled from @xml, and its copy for MJX with collisions restricted to
        task-relevant geom groups and parked objects held in place.
        """
        m0 = mujoco.MjModel.from_xml_string(xml)
        spec = mujoco.MjSpec.from_string(xml)
        for name, default in (("max_geom_pairs", _DEFAULT_MAX_GEOM_PAIRS),
                              ("max_contact_points", _DEFAULT_MAX_CONTACT_POINTS)):
            numeric = spec.add_numeric()
            numeric.name = name
            numeric.data = [int(self._mjx_config.get(name, default))]
        m = spec.compile()
        assert (m.nbody, m.ngeom, m.nq, m.nv, m.nu) == (m0.nbody, m0.ngeom, m0.nq, m0.nv, m0.nu)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name in self._parked_bodies:
                m.dof_damping[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = _PARKED_DAMPING
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            group = self._collision_group(m, m.geom_bodyid[g])
            contype, conaffinity = _COLLISION_MASKS[group] if group is not None else (0, 0)
            m.geom_contype[g] = contype
            m.geom_conaffinity[g] = conaffinity

        if "iterations" in self._mjx_config:
            m.opt.iterations = int(self._mjx_config["iterations"])
        if "ls_iterations" in self._mjx_config:
            m.opt.ls_iterations = int(self._mjx_config["ls_iterations"])
        if "tolerance" in self._mjx_config:
            m.opt.tolerance = float(self._mjx_config["tolerance"])
        if "cone" in self._mjx_config:
            m.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[self._mjx_config["cone"]]
        return m0, m

    def _models_for_xml(self, xml):
        if xml not in self._model_cache:
            m0, m = self._compile_mjx_model(xml)
            self._model_cache[xml] = (m0, m, mjx.put_model(m))
        return self._model_cache[xml]

    def _load_models(self, xmls):
        """Set the per-world models to those described by the list of model @xmls."""
        assert len(xmls) == self._num_envs
        if self._xmls == list(xmls):
            return
        self._restore_cpu_model()
        self._xmls = list(xmls)
        models = [self._models_for_xml(x) for x in xmls]
        mx_list = [mx for _, _, mx in models]
        if models[0][1].nhfield == 0:
            # static fields read only by heightfield collisions and the MJX-Warp backend
            base = mx_list[0]
            mx_list = [
                mx.replace(
                    geom_aabb=base.geom_aabb,
                    _impl=mx._impl.replace(geom_rbound_hfield=base._impl.geom_rbound_hfield),
                )
                for mx in mx_list
            ]

        leaves, treedef = jax.tree_util.tree_flatten(mx_list[0])
        leaves_per_world = [leaves]
        for mx in mx_list[1:]:
            world_leaves, world_treedef = jax.tree_util.tree_flatten(mx)
            if world_treedef != treedef:
                raise ValueError("per-world models differ in structure; they must share one MJX model layout")
            leaves_per_world.append(world_leaves)
        batched, axes = [], []
        for i in range(len(leaves)):
            values = [np.asarray(w[i]) for w in leaves_per_world]
            if all(np.array_equal(values[0], v) for v in values[1:]):
                batched.append(leaves[i])
                axes.append(None)
            else:
                batched.append(jnp.stack([jnp.asarray(v) for v in values]))
                axes.append(0)
        self._mx = jax.device_put(jax.tree_util.tree_unflatten(treedef, batched), self._device)
        self._mx_axes = jax.tree_util.tree_unflatten(treedef, axes)
        self._mx_axes_key = tuple(a is not None for a in axes)
        self._data_template = mjx.make_data(mx_list[0])
        self._m = models[0][1]
        self._state_dim = 1 + self._m.nq + self._m.nv

        # robosuite-side models, for evaluating observations of each world
        cpu_model = self._cpu.env.sim.model._model
        self._world_models = [m0 for m0, _, _ in models]
        fields = [f for f in _float_fields(cpu_model) if not f.startswith(_VISUAL_FIELD_PREFIXES)]
        self._world_fields = [
            [f for f in fields if not np.array_equal(getattr(m0, f), getattr(cpu_model, f))]
            for m0 in self._world_models
        ]
        self._cpu_defaults = {f: np.array(getattr(cpu_model, f)) for fs in self._world_fields for f in fs}

    # ---------------------------------------------------------------------------------
    # per-world JAX functions (vmapped over the batch)
    # ---------------------------------------------------------------------------------

    @staticmethod
    def _fwd_position_velocity(m, d):
        d = mjx.fwd_position(m, d)
        return mjx.fwd_velocity(m, d)

    @staticmethod
    def _fwd_dynamics_and_integrate(m, d):
        d = mjx.fwd_actuation(m, d)
        d = mjx.fwd_acceleration(m, d)
        if d._impl.efc_J.size == 0:
            d = d.replace(qacc=d.qacc_smooth)
        else:
            d = mjx_solver.solve(m, d)
        if m.opt.integrator == mjx.IntegratorType.EULER:
            return mjx.euler(m, d)
        if m.opt.integrator == mjx.IntegratorType.IMPLICITFAST:
            return mjx.implicit(m, d)
        raise NotImplementedError("integrator {} not supported in MJX mode".format(m.opt.integrator))

    def _reset_single(self, m, state):
        nq, nv = self._m.nq, self._m.nv
        d = self._data_template.replace(
            time=state[0],
            qpos=state[1:1 + nq],
            qvel=state[1 + nq:1 + nq + nv],
        )
        grip = jnp.zeros((len(self._arms), 2))
        q0 = tuple(d.qpos[arm.qpos] for arm in self._arms)
        return d, grip, q0

    def _step_single(self, m, d, grip, q0, action):
        goal_frame = mjx.kinematics(m, d)
        goals = [arm.goal(goal_frame, action) for arm in self._arms]
        grip = jnp.stack([arm.step_grip(grip[i], action) for i, arm in enumerate(self._arms)])
        grip_ctrls = [arm.grip_ctrl(grip[i]) for i, arm in enumerate(self._arms)]

        def substep(d, _):
            d = self._fwd_position_velocity(m, d)
            ctrl = d.ctrl
            for arm, (goal_pos, goal_ori), q0_arm, grip_ctrl in zip(self._arms, goals, q0, grip_ctrls):
                ctrl = ctrl.at[arm.arm_act].set(arm.torques(m, d, goal_pos, goal_ori, q0_arm))
                ctrl = ctrl.at[arm.grip_act].set(grip_ctrl)
            pre = (d.qpos, d.qvel)
            return self._fwd_dynamics_and_integrate(m, d.replace(ctrl=ctrl)), pre

        d, pre = jax.lax.scan(substep, d, None, length=self._n_substeps)
        pre = jax.tree.map(lambda x: x[-1], pre)
        diverged = ~(jnp.all(jnp.isfinite(d.qpos)) & jnp.all(jnp.isfinite(d.qvel)) & jnp.all(jnp.isfinite(d.ctrl)))
        return d, grip, pre, diverged

    def _jitted(self, name):
        key = (name, self._mx_axes_key)
        if key not in self._jit_cache:
            if name == "reset":
                fn = jax.vmap(self._reset_single, in_axes=(self._mx_axes, 0))
            else:
                fn = jax.vmap(self._step_single, in_axes=(self._mx_axes, 0, 0, 0, 0))
            fn = jax.jit(fn)

            def traced_with_collision_functions(*args):
                with _collision_functions(self._collision_funcs):
                    return fn(*args)

            self._jit_cache[key] = traced_with_collision_functions
        return self._jit_cache[key]

    # ---------------------------------------------------------------------------------
    # robosuite evaluation of each world
    # ---------------------------------------------------------------------------------

    def _restore_cpu_model(self):
        m = self._cpu.env.sim.model._model
        for f in self._cpu_defaults:
            getattr(m, f)[:] = self._cpu_defaults[f]

    def _load_world_into_cpu(self, i, time, qpos, qvel, ctrl=None, fresh_qpos=None, fresh_qvel=None):
        """
        Set the CPU robosuite simulation to world @i: run forward dynamics at (@qpos, @qvel, @ctrl), then
        overwrite positions and velocities with (@fresh_qpos, @fresh_qvel) if given.
        """
        sim = self._cpu.env.sim
        m, data = sim.model._model, sim.data._data
        self._restore_cpu_model()
        for f in self._world_fields[i]:
            getattr(m, f)[:] = getattr(self._world_models[i], f)
        data.time = time
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        if ctrl is not None:
            data.ctrl[:] = ctrl
        mujoco.mj_forward(m, data)
        if fresh_qpos is not None:
            data.qpos[:] = fresh_qpos
            data.qvel[:] = fresh_qvel

    def _evaluate_worlds(self, time, qpos, qvel, ctrl=None, fresh_qpos=None, fresh_qvel=None, action=None):
        obs, reward, success = [], [], []
        for i in range(self._num_envs):
            self._load_world_into_cpu(
                i, time[i], qpos[i], qvel[i],
                ctrl=None if ctrl is None else ctrl[i],
                fresh_qpos=None if fresh_qpos is None else fresh_qpos[i],
                fresh_qvel=None if fresh_qvel is None else fresh_qvel[i],
            )
            obs.append(self._cpu.get_observation())
            reward.append(self._cpu.env.reward(None if action is None else action[i]))
            success.append(self._cpu.is_success())
        self._obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        self._reward = np.asarray(reward, dtype=float)
        self._success = {k: np.asarray([s[k] for s in success], dtype=bool) for k in success[0]}

    # ---------------------------------------------------------------------------------
    # EnvBase API
    # ---------------------------------------------------------------------------------

    def _set_states(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = np.broadcast_to(states, (self._num_envs, states.shape[0]))
        assert states.shape == (self._num_envs, self._state_dim), \
            "expected states of shape {}, got {}".format((self._num_envs, self._state_dim), states.shape)
        self._d, self._grip, self._q0 = self._jitted("reset")(self._mx, jax.device_put(states, self._device))
        nq = self._m.nq
        self._evaluate_worlds(states[:, 0], states[:, 1:1 + nq], states[:, 1 + nq:])
        self._reward = np.zeros(self._num_envs)
        return deepcopy(self._obs)

    def sample_reset_states(self, n):
        """Initial simulator states drawn from the robosuite reset distribution."""
        self._restore_cpu_model()
        states = []
        for _ in range(n):
            self._cpu.env.reset()
            states.append(self._cpu.get_state()["states"])
        return np.stack(states)

    def step(self, action):
        """
        Step all environments with a batch of actions.

        Args:
            action (np.array): actions of shape (num_envs, action_dim)

        Returns:
            observation (dict): batched observation dictionary
            reward (np.array): rewards of shape (num_envs,)
            done (np.array): done flags of shape (num_envs,)
            info (dict): "is_success" (dict of (num_envs,) bool arrays) and "diverged" ((num_envs,) bool array)
        """
        action = np.asarray(action, dtype=np.float64).reshape(self._num_envs, -1)
        self._d, self._grip, pre, diverged = self._jitted("step")(
            self._mx, self._d, self._grip, self._q0, jax.device_put(action, self._device))
        d = self._d
        time, qpos, qvel, ctrl, pre_qpos, pre_qvel, diverged = jax.device_get(
            (d.time, d.qpos, d.qvel, d.ctrl, pre[0], pre[1], diverged))
        diverged = np.asarray(diverged)
        safe = lambda x: np.where(diverged[:, None], 0., x)
        self._evaluate_worlds(
            time, safe(pre_qpos), safe(pre_qvel), ctrl=safe(ctrl),
            fresh_qpos=safe(qpos), fresh_qvel=safe(qvel), action=action,
        )
        info = dict(is_success=self.is_success(), diverged=diverged)
        return deepcopy(self._obs), self._reward.copy(), self.is_done(), info

    def reset(self):
        """
        Reset all environments to fresh samples from the robosuite reset distribution.

        Returns:
            observation (dict): batched initial observation dictionary
        """
        if self._xmls != [self._cpu_xml] * self._num_envs:
            self._load_models([self._cpu_xml] * self._num_envs)
        return self._set_states(self.sample_reset_states(self._num_envs))

    def reset_to(self, state):
        """
        Reset to specific simulator states.

        Args:
            state (dict): contains:
                - states (np.ndarray): flattened robosuite state of shape (state_dim,) applied to every
                    environment, or of shape (num_envs, state_dim)
                - model (str or list of str): optional robosuite model xml, shared by every environment
                    or one per environment. All models must share one layout.

        Returns:
            observation (dict): batched observation dictionary after setting the simulator states
        """
        if "model" in state:
            raw_xmls = state["model"]
            if isinstance(raw_xmls, str):
                raw_xmls = [raw_xmls] * self._num_envs
            edited = {x: self._cpu.env.edit_model_xml(x) for x in set(raw_xmls)}
            xmls = [edited[x] for x in raw_xmls]
            if xmls[0] != self._cpu_xml:
                self._restore_cpu_model()
                self._cpu.reset_to({"model": raw_xmls[0]})
                self._cpu_xml = xmls[0]
                self._cpu_defaults = {}
                self._xmls = None
                self._setup_arms()
                self._jit_cache = {}
            self._load_models(xmls)
        return self._set_states(state["states"])

    def render(self, mode="rgb_array", height=None, width=None, camera_name="agentview", env_index=0):
        """Render batch member @env_index off-screen through the CPU robosuite environment."""
        assert mode == "rgb_array", "MJX mode only supports rgb_array rendering"
        state = self.get_state()["states"][env_index]
        nq = self._m.nq
        self._load_world_into_cpu(env_index, state[0], state[1:1 + nq], state[1 + nq:])
        return self._cpu.render(mode=mode, height=height, width=width, camera_name=camera_name)

    def get_observation(self):
        return deepcopy(self._obs)

    def get_state(self):
        d = self._d
        states = jnp.concatenate([d.time[:, None], d.qpos, d.qvel], axis=1)
        return dict(states=np.asarray(states))

    def get_reward(self):
        return self._reward.copy()

    def get_goal(self):
        raise NotImplementedError

    def set_goal(self, **kwargs):
        raise NotImplementedError

    def is_done(self):
        return np.zeros(self._num_envs, dtype=bool)

    def is_success(self):
        return {k: v.copy() for k, v in self._success.items()}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def action_dimension(self):
        return self._cpu.action_dimension

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return EB.EnvType.MJX_TYPE

    @property
    def version(self):
        return self._cpu.version

    def serialize(self):
        kwargs = deepcopy(self._init_kwargs)
        kwargs.update(num_envs=self._num_envs, mjx_config=deepcopy(self._mjx_config))
        return dict(env_name=self.name, env_version=self.version, type=self.type, env_kwargs=kwargs)

    @classmethod
    def create_for_data_processing(cls, *args, **kwargs):
        raise NotImplementedError("use EnvRobosuite for dataset processing")

    @property
    def rollout_exceptions(self):
        return ()

    @property
    def base_env(self):
        return self._cpu.base_env

    def __repr__(self):
        return self.name + " (MJX, num_envs={})\n".format(self._num_envs) + json.dumps(
            dict(env_kwargs=self._init_kwargs, mjx_config=self._mjx_config), sort_keys=True, indent=4, default=str
        )
