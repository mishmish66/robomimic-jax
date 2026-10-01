"""
Observations, success, and reward of the robomimic robosuite tasks as JAX functions of simulated world data.

Each task produces robosuite's "object" observation vector, sensor for sensor. Robosuite computes the
pose of an object relative to a gripper from the object pose cached by the previous observation pass;
tasks with such sensors keep that pose in a cache carried from one state to the next.
"""
import numpy as np

import jax.numpy as jnp

from robomimic.geometry import ContactTest, mat2quat, quat2mat, relative_pose, wxyz_to_xyzw


def _ids(view, names):
    return [view(n).id for n in names]


def _flag(x):
    return x[None].astype(jnp.float32)


def _body_pose_xyzw(d, body):
    return d.xpos[body], wxyz_to_xyzw(d.xquat[body])


def _relative_to_gripper(pos, quat, arm, d):
    """robosuite's `{obj}_to_{prefix}eef_pos` and `_quat`: object pose (`pos`, `quat`) in the gripper frame."""
    return relative_pose(pos, quat2mat(quat), arm.eef_pos(d), arm.eef_rot(d))


def _pose_obs(cache, poses, arm, d):
    """For each object: its cached pose relative to the gripper of `arm`, then its current position and quaternion."""
    parts = []
    for (cached_pos, cached_quat), (pos, quat) in zip(cache, poses):
        rel_pos, rel_quat = _relative_to_gripper(cached_pos, cached_quat, arm, d)
        parts += [rel_pos, rel_quat, pos, quat]
    return parts


def _opposite_sides(a, b, axis):
    """Whether the components of `a` and `b` perpendicular to `axis` point more than 90 degrees apart."""
    return jnp.dot(jnp.cross(a, axis), jnp.cross(b, axis)) < 0


def _reach_is_far(gripper_pos, obj_pos):
    """robosuite's `1 - tanh(10 * dist) < 0.6`: the gripper has let go of the object."""
    return 1. - jnp.tanh(10. * jnp.linalg.norm(gripper_pos - obj_pos)) < 0.6


class Task:
    """Base class; tasks without cached sensors return an empty cache."""
    def cache(self, d):
        return ()


class Lift(Task):
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.cube = m.body(spec["cube"]).id
        self.table_height = spec["table_height"]
        self.reward_scale = spec["reward_scale"]

    def object_obs(self, d, cache):
        cube_pos, cube_quat = _body_pose_xyzw(d, self.cube)
        return jnp.concatenate([cube_pos, cube_quat, cube_pos - self.arm.eef_pos(d)])

    def success(self, d):
        return d.xpos[self.cube, 2] > self.table_height + 0.04

    def reward(self, d):
        return self.success(d).astype(jnp.float32) * 2.25 * self.reward_scale


class _PlacementTask(Task):
    """Tasks that place objects (PickPlace: into bins, NutAssembly: onto pegs) and observe them relative to the gripper."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.bodies = _ids(m.body, spec["bodies"])
        self.observed = spec["observed"]
        self.single = spec["single"]
        self.reward_scale = spec["reward_scale"]

    def cache(self, d):
        return tuple(_body_pose_xyzw(d, self.bodies[i]) for i in self.observed)

    def object_obs(self, d, cache):
        poses = [_body_pose_xyzw(d, self.bodies[i]) for i in self.observed]
        return jnp.concatenate(_pose_obs(cache, poses, self.arm, d))

    def _placed(self, d, i):
        raise NotImplementedError

    def _placed_all(self, d):
        return jnp.stack([
            self._placed(d, i) & _reach_is_far(self.arm.eef_pos(d), d.xpos[body]) for i, body in enumerate(self.bodies)
        ])

    def success(self, d):
        placed = self._placed_all(d)
        return jnp.any(placed) if self.single else jnp.all(placed)

    def reward(self, d):
        return jnp.sum(self._placed_all(d)).astype(jnp.float32) * self.reward_scale


class PickPlace(_PlacementTask):
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.bin_low = np.asarray(spec["bin_low"])
        self.bin_high = np.asarray(spec["bin_high"])
        self.bin_z = spec["bin_z"]

    def _placed(self, d, i):
        p = d.xpos[self.bodies[i]]
        return (
            jnp.all((self.bin_low[i] < p[:2]) & (p[:2] < self.bin_high[i]))
            & (self.bin_z < p[2]) & (p[2] < self.bin_z + 0.1)
        )


class NutAssembly(_PlacementTask):
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.pegs = _ids(m.body, spec["pegs"])
        self.table_height = spec["table_height"]

    def _placed(self, d, i):
        p, peg = d.xpos[self.bodies[i]], d.xpos[self.pegs[i]]
        return jnp.all(jnp.abs(p[:2] - peg[:2]) < 0.03) & (p[2] < self.table_height + 0.05)


class ToolHang(Task):
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.base_geom = m.geom(spec["base_geom"]).id
        self.frame_site = m.site(spec["frame_site"]).id
        self.tool_body = m.body(spec["tool_body"]).id
        self.stand_mount_site = m.site(spec["stand_mount_site"]).id
        self.frame_tip_site = m.site(spec["frame_tip_site"]).id
        self.frame_mount_site = m.site(spec["frame_mount_site"]).id
        self.frame_hang_site = m.site(spec["frame_hang_site"]).id
        self.stand_walls = _ids(m.geom, spec["stand_walls"])
        self.tool_hole_center = m.site(spec["tool_hole_center"]).id
        self.hole_geoms = _ids(m.geom, spec["hole_geoms"])
        self.hole_tolerance = spec["hole_tolerance"]
        tool_geoms = _ids(m.geom, spec["tool_geoms"])
        self.robot_tool_contact = ContactTest(m, _ids(m.geom, spec["robot_geoms"]), tool_geoms)
        self.hole_hook_contact = ContactTest(m, _ids(m.geom, spec["hole_ring_geoms"]), _ids(m.geom, spec["hook_geoms"]))
        self.reward_scale = spec["reward_scale"]

    def cache(self, d):
        return (
            (d.geom_xpos[self.base_geom], mat2quat(d.geom_xmat[self.base_geom].reshape(3, 3))),
            (d.site_xpos[self.frame_site], mat2quat(d.site_xmat[self.frame_site].reshape(3, 3))),
            (d.xpos[self.tool_body], mat2quat(d.xmat[self.tool_body].reshape(3, 3))),
        )

    def _frame_assembled(self, d):
        base_pos = d.geom_xpos[self.base_geom]
        shaft = d.site_xpos[self.stand_mount_site] - base_pos
        shaft = shaft / jnp.linalg.norm(shaft)
        vertical = jnp.arccos(shaft[2]) < np.pi / 18.
        close = jnp.linalg.norm(d.site_xpos[self.frame_tip_site] - base_pos) < 0.05
        hook_end = d.site_xpos[self.frame_mount_site]
        hook = d.site_xpos[self.frame_site] - hook_end
        hook = hook / jnp.linalg.norm(hook)
        walls = [d.geom_xpos[g] - hook_end for g in self.stand_walls]
        between = _opposite_sides(walls[0], walls[2], hook) & _opposite_sides(walls[1], walls[3], hook)
        return vertical & close & between

    def _tool_on_frame(self, d):
        hook_end = d.site_xpos[self.frame_hang_site]
        hook = d.site_xpos[self.frame_site] - hook_end
        hook_length = jnp.linalg.norm(hook)
        hook = hook / hook_length
        hole = d.site_xpos[self.tool_hole_center] - hook_end
        along = jnp.dot(hole, hook)
        close = jnp.linalg.norm(hole - along * hook) < self.hole_tolerance
        g1, g2 = (d.geom_xpos[g] - hook_end for g in self.hole_geoms)
        between = _opposite_sides(g1, g2, hook)
        inserted = (along / hook_length > 0.05) & (along / hook_length < 1.)
        return ~self.robot_tool_contact(d) & self.hole_hook_contact(d) & close & between & inserted

    def object_obs(self, d, cache):
        parts = _pose_obs(cache, self.cache(d), self.arm, d)
        return jnp.concatenate(parts + [_flag(self._frame_assembled(d)), _flag(self._tool_on_frame(d))])

    def success(self, d):
        return self._frame_assembled(d) & self._tool_on_frame(d)

    def reward(self, d):
        return self.success(d).astype(jnp.float32) * self.reward_scale


class TwoArmTransport(Task):
    def __init__(self, spec, m, arms):
        self.arms = arms
        self.payload, self.trash = m.body(spec["payload"]).id, m.body(spec["trash"]).id
        self.lid_handle = m.geom(spec["lid_handle"]).id
        self.target_bin, self.trash_bin = m.geom(spec["target_bin"]).id, m.geom(spec["trash_bin"]).id
        self.payload_in_target_bin = ContactTest(m, _ids(m.geom, spec["target_bin_geoms"]), _ids(m.geom, spec["payload_geoms"]))
        self.trash_in_trash_bin = ContactTest(m, _ids(m.geom, spec["trash_bin_geoms"]), _ids(m.geom, spec["trash_geoms"]))
        self.reward_scale = spec["reward_scale"]

    def object_obs(self, d, cache):
        payload_pos, trash_pos, lid_pos = d.xpos[self.payload], d.xpos[self.trash], d.geom_xpos[self.lid_handle]
        g0, g1 = (arm.eef_pos(d) for arm in self.arms)
        return jnp.concatenate([
            payload_pos, mat2quat(d.xmat[self.payload].reshape(3, 3)),
            trash_pos, mat2quat(d.xmat[self.trash].reshape(3, 3)),
            lid_pos, mat2quat(d.geom_xmat[self.lid_handle].reshape(3, 3)),
            d.geom_xpos[self.target_bin], d.geom_xpos[self.trash_bin],
            _flag(self.payload_in_target_bin(d)), _flag(self.trash_in_trash_bin(d)),
            payload_pos - g0, lid_pos - g0, payload_pos - g1, trash_pos - g1,
        ])

    def success(self, d):
        return self.payload_in_target_bin(d) & self.trash_in_trash_bin(d)

    def reward(self, d):
        return self.success(d).astype(jnp.float32) * self.reward_scale


TASKS = {
    "lift": Lift,
    "can": PickPlace,
    "square": NutAssembly,
    "tool_hang": ToolHang,
    "transport": TwoArmTransport,
}
