"""
Observations, success, and rewards of the robomimic robosuite tasks as JAX functions of simulated world data.

Each task produces robosuite's "object" observation vector, sensor for sensor. Robosuite computes the
pose of an object relative to a gripper from the object pose cached by the previous observation pass;
tasks with such sensors keep that pose in a cache carried from one state to the next.
"""
import numpy as np

import jax.numpy as jnp

from robomimic.env.geometry import ContactTest, mat2quat, quat2mat, relative_pose, wxyz_to_xyzw


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


# robosuite's staged reward multipliers of the placement tasks
_REACH, _GRASP, _LIFT, _HOVER = 0.1, 0.35, 0.5, 0.7


def _closeness(dist, rate=10.):
    """robosuite's `1 - tanh(rate * dist)`."""
    return 1. - jnp.tanh(rate * dist)


class _Grasp:
    """robosuite's `_check_grasp`: every finger pad of a gripper touches one of a set of objects."""
    def __init__(self, m, pad_geoms, object_geoms):
        self.tests = [[ContactTest(m, _ids(m.geom, pad), _ids(m.geom, geoms)) for geoms in object_geoms] for pad in pad_geoms]

    def __call__(self, d, objects=None):
        """Whether each finger pad touches one of the objects selected by boolean mask `objects` (default: all)."""
        touching = jnp.array([[test(d) for test in pad] for pad in self.tests])
        if objects is not None:
            touching = touching & objects
        return jnp.all(jnp.any(touching, axis=1))


class Task:
    """Base class; tasks without cached sensors return an empty cache."""
    def cache(self, d):
        return ()

    def shaped_reward(self, d):
        """robosuite's shaped reward; robosuite has none for this task, so the sparse reward."""
        return self.reward(d)


class Lift(Task):
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.cube = m.body(spec["cube"]).id
        self.table_height = spec["table_height"]
        self.reward_scale = spec["reward_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["cube_geoms"]])

    def object_obs(self, d, cache):
        cube_pos, cube_quat = _body_pose_xyzw(d, self.cube)
        return jnp.concatenate([cube_pos, cube_quat, cube_pos - self.arm.eef_pos(d)])

    def success(self, d):
        return d.xpos[self.cube, 2] > self.table_height + 0.04

    def reward(self, d):
        return self.success(d).astype(jnp.float32) * 2.25 * self.reward_scale

    def shaped_reward(self, d):
        """2.25 on success, else reaching the cube (in [0, 1]) plus 0.25 while grasping it, scaled like the sparse reward."""
        reach = _closeness(jnp.linalg.norm(self.arm.eef_pos(d) - d.xpos[self.cube]))
        return jnp.where(self.success(d), 2.25, reach + 0.25 * self.grasp(d)) * self.reward_scale


class _PlacementTask(Task):
    """Tasks that place objects (PickPlace: into bins, NutAssembly: onto pegs) and observe them relative to the gripper."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.bodies = _ids(m.body, spec["bodies"])
        self.observed = spec["observed"]
        self.single = spec["single"]
        self.reward_scale = spec["reward_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], spec["object_geoms"])

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

    def _reach_targets(self, d):
        """Positions the gripper reaches for, one per object."""
        raise NotImplementedError

    def _lift_height(self, d):
        raise NotImplementedError

    def _hover(self, d, r_lift):
        """Hovering reward of each object."""
        raise NotImplementedError

    def _staged(self, d, active):
        """robosuite's reaching, grasping, lifting, and hovering rewards of the objects selected by mask `active`."""
        reach = jnp.linalg.norm(self._reach_targets(d) - self.arm.eef_pos(d), axis=1)
        r_reach = _closeness(jnp.min(jnp.where(active, reach, jnp.inf))) * _REACH
        grasped = self.grasp(d, active)
        r_grasp = grasped * _GRASP
        below = jnp.maximum(self._lift_height(d) - d.xpos[np.asarray(self.bodies), 2], 0.)
        lifted = _GRASP + _closeness(jnp.min(jnp.where(active, below, jnp.inf)), 15.) * (_LIFT - _GRASP)
        r_lift = jnp.where(grasped, lifted, 0.)
        r_hover = jnp.max(jnp.where(active, self._hover(d, r_lift), 0.))
        return jnp.stack([r_reach, r_grasp, r_lift, r_hover])

    def shaped_reward(self, d):
        """
        1 per placed object plus the largest staged reward of the unplaced ones, scaled like the sparse reward.
        Objects parked outside the workspace count as unplaced, as in robosuite.
        """
        placed = self._placed_all(d)
        return (jnp.sum(placed) + jnp.max(self._staged(d, ~placed))) * self.reward_scale


class PickPlace(_PlacementTask):
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.bin_low = np.asarray(spec["bin_low"])
        self.bin_high = np.asarray(spec["bin_high"])
        self.bin_z = spec["bin_z"]
        self.bin_centers = np.asarray(spec["bin_centers"])
        self.bin_size = np.asarray(spec["bin_size"])

    def _reach_targets(self, d):
        return d.xpos[np.asarray(self.bodies)]

    def _lift_height(self, d):
        return self.bin_z + 0.25

    def _hover(self, d, r_lift):
        """Objects above their bins earn the full lifting reward, others `r_lift`, plus closeness to the bin center."""
        offset = d.xpos[np.asarray(self.bodies), :2] - self.bin_centers
        above = jnp.all(jnp.abs(offset) < self.bin_size / 4., axis=1)
        return jnp.where(above, _LIFT, r_lift) + _closeness(jnp.linalg.norm(offset, axis=1)) * (_HOVER - _LIFT)

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
        self.handles = _ids(m.site, spec["handles"])
        self.table = m.body(spec["table"]).id

    def _reach_targets(self, d):
        return d.site_xpos[np.asarray(self.handles)]

    def _lift_height(self, d):
        return d.xpos[self.table, 2] + 0.2

    def _hover(self, d, r_lift):
        offset = d.xpos[np.asarray(self.bodies), :2] - d.xpos[np.asarray(self.pegs), :2]
        return r_lift + _closeness(jnp.linalg.norm(offset, axis=1)) * (_HOVER - _LIFT)

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
