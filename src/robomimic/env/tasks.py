"""
Observations, success, and rewards of the robomimic and MimicGen robosuite tasks as JAX functions of simulated world
data.

Each task produces robosuite's "object" observation vector, sensor for sensor. Robosuite computes the
pose of an object relative to a gripper from the object pose cached by the previous observation pass;
tasks with such sensors keep that pose in a cache carried from one state to the next.
"""
import mujoco
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


# distance scale of reaching; each task's carry scales cover the farthest its objects start from their goals
_REACH_SCALE = 0.5
# height the gripper approaches an object from, and horizontal distances within which a carried object and a
# reaching gripper descend
_APPROACH_HEIGHT, _ALIGN, _REACH_ALIGN = 0.15, 0.02, 0.05
# distance over which a nut closes in on a pose its peg fits through
_FINE_ALIGN = 0.08
# height above a peg's tip over which a nut lowered onto the peg closes in on the tip
_TIP_APPROACH = 0.3
# height above its clearance a carried object may rise freely, and the further height over which lifting stops counting
_HEIGHT_SLACK, _OVERSHOOT = 0.05, 0.3
# clearance shortfall over which a nut's fit counts finely, and tilt (1 - cos) over which its levelness counts coarsely
# and finely
_FIT_FINE, _LEVEL_SCALE, _LEVEL_FINE = 0.01, 1., 0.1
# horizontal distance from the target over which a carried object descends from clearance to placing height
_DESCENT_BAND = 0.05
# distance over which the arm taking an object over from the other closes in on it
_HANDOVER_REACH = 1.
# distance from each end of the payload's handle that a gripper reaching along it stops short of
_HANDLE_END = 0.02
# alignment (cosine) of a gripper's fingers with the payload handle's long axis up to which a grip counts fully, and the
# further alignment over which it stops counting
_ACROSS_SLACK, _ACROSS = 0.3, 0.4
# offset of the frame's tip from the slot over which its centering counts finely
_SLOT_FINE = 0.01
# distance from its limits within which an arm joint leaves the arm less room to turn the frame
_JOINT_ROOM = np.radians(15.)
# angle over which the frame's turn to its assembled orientation counts finely
_FRAME_YAW_FINE = np.radians(10.)
# distance of the gripper from a released object at which it has let go (robosuite's far-gripper test passes at 4.2 cm)
_LET_GO = 0.05
# height above its placing height within which a released object counts as placed
_RELEASE_HEIGHT = 0.25
# coarse distance scale of closing in vertically, covering the workspace's height
_FAR = 1.


def _half_height(m, geom):
    """Half the height of a box or cylinder geom of model `m`, 0 for other types."""
    if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        return float(m.geom_size[geom, 2])
    if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(m.geom_size[geom, 1])
    return 0.


def _ramp(dist, scale):
    """1 at `dist` 0, falling linearly to 0 at `scale`."""
    return 1. - jnp.clip(dist / scale, 0., 1.)


# alignment of the gripper axis with straight down above which the gripper counts as upright (about 20 degrees)
_UPRIGHT = 0.94


def _upright(arm, d):
    """1 while the gripper points down within `_UPRIGHT`, falling linearly to 0 as it turns horizontal."""
    return jnp.clip(-arm.eef_rot(d)[2, 2] / _UPRIGHT, 0., 1.)


def _fine(dist, scale, fine):
    """1 at `dist` 0: half linearly over `scale`, half linearly over the shorter `fine`."""
    return 0.5 * _ramp(dist, scale) + 0.5 * _ramp(dist, fine)


def _closing(dist, scale):
    """1 at `dist` 0, falling to 0 at `_FAR`: half linearly over `scale`, half linearly over `_FAR`."""
    return 0.5 * _ramp(dist, scale) + 0.5 * _ramp(dist, _FAR)


def _finger_alignment(grasp, d, grip, axis):
    """
    From 0.5 with the fingers of `grasp` (first pad to second) pointing against signed axis `axis` (1-based: -2 is minus
    y) of geom `grip`, through 0.75 across it, to 1 along it, as the demonstrations grasp.
    """
    a, b = d.geom_xpos[grasp.pads[0]], d.geom_xpos[grasp.pads[1]]
    fingers = (b - a) / jnp.maximum(jnp.linalg.norm(b - a), 1e-6)
    direction = np.sign(axis) * d.geom_xmat[grip].reshape(3, 3)[:, abs(axis) - 1]
    return 0.75 + 0.25 * (fingers @ direction)


def _tilt(upright):
    """Scale from 0.5 for a gripper pointing sideways to 1 for an upright one."""
    return 0.5 + 0.5 * upright


def _approach(offset, height, scale, drop, align=_ALIGN):
    """
    Closeness in [0, 1] of approaching a point from above, at horizontal distance `offset` and `height` above it:
    half from horizontal closeness (`_closing` over `scale`), half from descending (`_closing` over `drop`) once within
    `align` of it.
    """
    return 0.5 * _closing(offset, scale) + 0.5 * _ramp(offset, align) * _closing(height, drop)


def _reach(arm, d, target):
    """Closeness of the gripper approaching `target` from above."""
    eef = arm.eef_pos(d)
    return _approach(
        jnp.linalg.norm(eef[:2] - target[:2]), jnp.abs(eef[2] - target[2]), _REACH_SCALE, _APPROACH_HEIGHT, _REACH_ALIGN,
    )


def _grip(arm, grasp, d, held, target, objects=None):
    """
    1 while `held`, else half of how far the gripper has closed with one of `objects` between its fingers
    (`_Grasp.enclosed` at grip point `target`).
    """
    enclosed = grasp.enclosed(d, arm.eef_pos(d), target, objects)
    return jnp.where(held, 1., 0.5 * enclosed * (1. - arm.grip_opening(d)))


def _lift_and_carry(z, offset, place, clearance, scale):
    """
    Closeness of the three carrying stages of an object at height `z` and horizontal distance `offset` from its target:
    lifting it to `clearance` while away from the target, which counts less once it rises over `_HEIGHT_SLACK` above
    `clearance`, moving it over the target (half `_closing` over `scale`, half over the last `_ALIGN`), which counts
    only as far as it is lifted, and
    lowering it to its placing height `place`, from above or below, once within `_ALIGN` of it.
    """
    lift_height = clearance - place
    goal_z = place + lift_height * jnp.clip(offset / _DESCENT_BAND, 0., 1.)
    up = _ramp(jnp.maximum(goal_z - z, 0.), lift_height) * _ramp(jnp.maximum(z - clearance - _HEIGHT_SLACK, 0.), _OVERSHOOT)
    over = up * (0.5 * _closing(offset, scale) + 0.5 * _ramp(offset, _ALIGN))
    down = _ramp(offset, _ALIGN) * _closing(jnp.abs(z - place), lift_height)
    return up, over, down


def _progress(reach, grip, held, *carry, carry_weight=0.6):
    """
    Progress in [0, 1] from reaching closeness, gripping (`_grip`), and, while the object is held, closeness of each
    carrying stage, which together weigh `carry_weight` and reaching and gripping half the rest each.
    """
    rest = 0.5 * (1. - carry_weight)
    return rest * reach + rest * grip + carry_weight * held * sum(carry) / len(carry)


# distance within which a finger pad counts as holding an object, so holds do not flicker as contacts separate
_HOLD_SLACK = 0.003
# distance of the gripper from an object's grip point or geoms within which closing on it counts toward gripping it
_ENCLOSE = 0.05


class _Grasp:
    """robosuite's `_check_grasp`: every finger pad of a gripper touches one of a set of objects."""
    def __init__(self, m, pad_geoms, object_geoms):
        self.tests = [[ContactTest(m, _ids(m.geom, pad), _ids(m.geom, geoms)) for geoms in object_geoms] for pad in pad_geoms]
        self.pads = [_ids(m.geom, pad)[0] for pad in pad_geoms]
        self.objects = [np.asarray(_ids(m.geom, geoms)) for geoms in object_geoms]

    def __call__(self, d, objects=None):
        """Whether each finger pad touches one of the objects selected by boolean mask `objects` (default: all)."""
        touching = jnp.array([[test(d) for test in pad] for pad in self.tests])
        if objects is not None:
            touching = touching & objects
        return jnp.all(jnp.any(touching, axis=1))

    def _between(self, d):
        """For each object, whether one of its geoms lies between the finger pads along the line joining them."""
        a, b = d.geom_xpos[self.pads[0]], d.geom_xpos[self.pads[1]]
        span = jnp.linalg.norm(b - a)
        axis = (b - a) / jnp.maximum(span, 1e-6)
        return jnp.array([jnp.any(((d.geom_xpos[g] - a) @ axis > 0.) & ((d.geom_xpos[g] - a) @ axis < span)) for g in self.objects])

    def enclosed(self, d, eef, target, objects=None):
        """
        Whether one of the objects selected by `objects` lies between the fingers, with the gripper at `eef` within
        `_ENCLOSE` of grip point `target` or of one of the object's geoms.
        """
        near = jnp.array([
            jnp.minimum(jnp.linalg.norm(eef - target), jnp.min(jnp.linalg.norm(d.geom_xpos[g] - eef, axis=-1))) < _ENCLOSE
            for g in self.objects
        ])
        inside = self._between(d) & near
        return jnp.any(inside & objects) if objects is not None else jnp.any(inside)

    def held(self, d, objects=None):
        """
        Whether the gripper holds one of the objects selected by `objects`: each finger pad is within `_HOLD_SLACK`
        of it, and one of its geoms lies between the pads along the line joining them.
        """
        touching = jnp.array([[test(d, _HOLD_SLACK) for test in pad] for pad in self.tests]) & self._between(d)
        if objects is not None:
            touching = touching & objects
        return jnp.any(jnp.all(touching, axis=0))


class Task:
    """
    Base class; tasks without cached sensors return an empty cache, and tasks that remember nothing of an episode
    beyond its current world have an empty memory. The success, reward and progress of a task with a memory take it
    after the world, unpacked.
    """
    def cache(self, d):
        return ()

    def initial_memory(self):
        """The memory at the start of an episode."""
        return ()

    def remember(self, d, memory):
        """`memory` updated with world `d`."""
        return memory

    def shaped_reward(self, d, *memory):
        """robosuite's shaped reward; robosuite has none for this task, so the sparse reward."""
        return self.reward(d, *memory)

    def extra_obs(self, d):
        """Low-dim observations of the task beyond "object" and the robots' proprioception."""
        return {}

    def progress(self, d):
        """Progress toward success in [0, 1]."""
        raise NotImplementedError

    def _potential(self, d, memory=()):
        """Progress, counting 1 once successful."""
        return jnp.where(self.success(d, *memory), 1., self.progress(d, *memory))

    def dense_reward(self, d, d_next, memory=(), memory_next=()):
        """1 on success after a step from world `d` to `d_next`, plus the change in progress."""
        success = self.success(d_next, *memory_next).astype(jnp.float32)
        return success + self._potential(d_next, memory_next) - self._potential(d, memory)


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

    def progress(self, d):
        """Approaching the cube from above, grasping it, and raising it to the success height, with the gripper upright."""
        held = self.grasp.held(d)
        reach = _reach(self.arm, d, d.xpos[self.cube])
        below = jnp.maximum(self.table_height + 0.04 - d.xpos[self.cube, 2], 0.)
        grip = _grip(self.arm, self.grasp, d, held, d.xpos[self.cube])
        return _progress(reach, grip, held, _ramp(below, 0.02)) * _upright(self.arm, d)


class _PlacementTask(Task):
    """Tasks that place objects (PickPlace: into bins, NutAssembly: onto pegs) and observe them relative to the gripper."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.bodies = _ids(m.body, spec["bodies"])
        self.observed = spec["observed"]
        self.single = spec["single"]
        self.reward_scale = spec["reward_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], spec["object_geoms"])
        self.carry_scale = spec["carry_scale"]

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

    def _target_offset(self, d, i):
        """Horizontal distance of object `i` from where it is placed, 0 over it."""
        raise NotImplementedError

    def _clearance(self, d):
        """Height a carried object's position clears the obstacles between it and its target at."""
        raise NotImplementedError

    def _place_height(self, d):
        """Height of a placed object's position."""
        raise NotImplementedError

    def progress(self, d):
        """
        Progress of the observed object: approaching it from above, grasping it, lifting it to clearance height,
        carrying it over its target and down onto it, and letting go of it, with the gripper upright. An object over its
        target, within `_RELEASE_HEIGHT` above its placing height, counts as held and reached once released.
        """
        i = self.observed[0]
        pos = d.xpos[self.bodies[i]]
        grasped = self.grasp.held(d, np.arange(len(self.bodies)) == i)
        offset = self._target_offset(d, i)
        carry = self._carry(d, i, pos, offset)
        height = pos[2] - self._place_height(d)
        released = (offset == 0.) & ~grasped & (height > -_ALIGN) & (height < _RELEASE_HEIGHT) & self._releasable(d, i, pos)
        reach = jnp.where(released, 1., _reach(self.arm, d, self._reach_targets(d)[i]))
        grip = _grip(self.arm, self.grasp, d, grasped | released, self._reach_targets(d)[i], np.arange(len(self.bodies)) == i)
        let_go = jnp.where(released, jnp.clip(jnp.linalg.norm(self.arm.eef_pos(d) - pos) / _LET_GO, 0., 1.), 0.)
        return _progress(reach, grip, grasped | released, *carry, let_go) * _upright(self.arm, d)

    def _releasable(self, d, i, pos):
        """Whether object `i` at `pos`, over its target, can be let go of there."""
        return True

    def _carry(self, d, i, pos, offset):
        """Closeness of each carrying stage of object `i` at `pos`, at horizontal distance `offset` from its target."""
        return _lift_and_carry(pos[2], offset, self._place_height(d), self._clearance(d), self.carry_scale)


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

    def _target_offset(self, d, i):
        """Distance from the object's bin shrunk by 3 cm on each side."""
        p = d.xpos[self.bodies[i], :2]
        return jnp.linalg.norm(p - jnp.clip(p, self.bin_low[i] + 0.03, self.bin_high[i] - 0.03))

    def _clearance(self, d):
        return self.bin_z + 0.2

    def _place_height(self, d):
        return self.bin_z + 0.06


class NutAssembly(_PlacementTask):
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.pegs = _ids(m.body, spec["pegs"])
        self.table_height = spec["table_height"]
        self.handles = _ids(m.site, spec["handles"])
        self.table = m.body(spec["table"]).id
        self.hole_half_widths, self.peg_half_widths = spec["hole_half_widths"], spec["peg_half_widths"]
        self.peg_half_heights = [max(_half_height(m, g) for g in range(m.ngeom) if m.geom_bodyid[g] == b) for b in self.pegs]

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

    def _fit_margin(self, d, i):
        """
        How far, in the nut's frame, its square hole clears the peg's axis at the nut's height, at their current yaws:
        positive once the peg fits through it.
        """
        nut, peg = d.xmat[self.bodies[i]].reshape(3, 3), d.xmat[self.pegs[i]].reshape(3, 3)
        yaw = jnp.arctan2(nut[1, 0], nut[0, 0]) - jnp.arctan2(peg[1, 0], peg[0, 0])
        peg_half = self.peg_half_widths[i] * (jnp.abs(jnp.cos(yaw)) + jnp.abs(jnp.sin(yaw)))
        on_axis = d.xpos[self.pegs[i]].at[2].set(d.xpos[self.bodies[i], 2])
        off = (nut.T @ (on_axis - d.xpos[self.bodies[i]]))[:2]
        return self.hole_half_widths[i] - peg_half - jnp.max(jnp.abs(off))

    def _carry(self, d, i, pos, offset):
        """
        Three stages: lifting the nut; moving it over its peg and seating it (level, and centered and turned so the peg fits through its hole), in equal parts,
        both counting only as far as it is lifted; and lowering it, half to the peg's tip and half down the peg, which the
        nut can only reach once the peg passes through its hole.
        """
        if self.hole_half_widths[i] is None:
            return super()._carry(d, i, pos, offset)
        place, clearance = self._place_height(d), self._clearance(d)
        up, over, _ = _lift_and_carry(pos[2], offset, place, clearance, self.carry_scale)
        level = jnp.clip(d.xmat[self.bodies[i]].reshape(3, 3)[2, 2], 0., 1.)
        margin = self._fit_margin(d, i)
        seated = _fine(1. - level, _LEVEL_SCALE, _LEVEL_FINE) * _fine(jnp.maximum(-margin, 0.), _FINE_ALIGN, _FIT_FINE)
        tip = d.xpos[self.pegs[i], 2] + self.peg_half_heights[i]
        to_tip = _ramp(offset, _ALIGN) * _closing(jnp.maximum(pos[2] - tip, 0.), _TIP_APPROACH)
        down = _ramp(offset, _ALIGN) * _ramp(jnp.maximum(pos[2] - place, 0.), tip - place)
        return up, 0.5 * (over + up * seated), 0.5 * (to_tip + down)

    def _releasable(self, d, i, pos):
        """A nut can be let go of over its peg once the peg fits through it, or once it is on the peg below its tip."""
        if self.hole_half_widths[i] is None:
            return True
        level = d.xmat[self.bodies[i]].reshape(3, 3)[2, 2]
        tip = d.xpos[self.pegs[i], 2] + self.peg_half_heights[i]
        return ((self._fit_margin(d, i) > 0.) & (level > 0.95)) | (pos[2] < tip)

    def _target_offset(self, d, i):
        """Distance from within 1.5 cm of the nut's peg."""
        return jnp.maximum(jnp.linalg.norm(d.xpos[self.bodies[i], :2] - d.xpos[self.pegs[i], :2]) - 0.015, 0.)

    def _clearance(self, d):
        return self.table_height + 0.18

    def _place_height(self, d):
        return self.table_height + 0.01


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
        self.hole_ring = _ids(m.geom, spec["hole_ring_geoms"])
        self.reward_scale = spec["reward_scale"]
        self.frame_grip, self.tool_grip = m.geom(spec["frame_grip"]).id, m.geom(spec["tool_grip"]).id
        self.frame_body = int(m.geom_bodyid[self.frame_grip])
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["frame_geoms"], spec["tool_grasp_geoms"]])
        self.frame_carry_scale, self.tool_carry_scale = spec["frame_carry_scale"], spec["tool_carry_scale"]
        joints = [int(np.flatnonzero(m.jnt_qposadr == a)[0]) for a in np.asarray(self.arm.qpos)]
        self.joint_lo, self.joint_hi = (jnp.asarray(r, jnp.float32) for r in m.jnt_range[joints].T)

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

    def _grip_alignment(self, d, grip, axis):
        """`_finger_alignment` of the fingers with signed axis `axis` (1-based) of geom `grip`."""
        return _finger_alignment(self.grasp, d, grip, axis)

    def _frame_yaw_error(self, d):
        """Angle from the frame turned, about the vertical, as the demonstrations insert it between the stand's walls."""
        R = d.xmat[self.frame_body].reshape(3, 3)
        walls = d.geom_xpos[np.asarray(self.stand_walls)]
        across = walls[0] - walls[2]
        error = jnp.arctan2(R[1, 0], R[0, 0]) - jnp.arctan2(across[1], across[0]) - np.pi
        return jnp.abs(jnp.mod(error + np.pi, 2 * np.pi) - np.pi)

    def _joint_room(self, d):
        """1 while every arm joint is at least `_JOINT_ROOM` from its limits, falling to 0 as one reaches its limit."""
        q = d.qpos[np.asarray(self.arm.qpos)]
        room = jnp.minimum(q - self.joint_lo, self.joint_hi - q)
        return jnp.min(_ramp(jnp.maximum(_JOINT_ROOM - room, 0.), _JOINT_ROOM))

    def _frame_turn_error(self, d):
        """Angle of the rotation from the frame's orientation to its post upright and turned to fit between the walls."""
        R = d.xmat[self.frame_body].reshape(3, 3)
        walls = d.geom_xpos[np.asarray(self.stand_walls)]
        across = (walls[2] - walls[0]) * jnp.array([1., 1., 0.])
        x = across / jnp.linalg.norm(across)
        z = jnp.array([0., 0., 1.])
        target = jnp.stack([x, jnp.cross(z, x), z], axis=1)
        return jnp.arccos(jnp.clip(0.5 * (jnp.trace(target.T @ R) - 1.), -1., 1.))

    def _frame_progress(self, d):
        """
        Reaching the frame's grip, holding the frame, turning its post upright, turning the frame, coarsely and finely,
        toward its post upright and turned to fit between the stand's walls, lifting its tip over them as far as the post is upright, bringing it over the
        slot between them, half of which counts only as far as it is turned to fit, lowering it into the slot as far as
        it is turned to fit, centering it over the slot finely (`_SLOT_FINE`) as far as it is turned to fit, and
        inserting it down to the slot's floor, linearly from the walls' height, as far as it is so centered, all counting as far as the arm's joints keep room from their limits (`_joint_room`); held and reached once
        released upright over the slot, and 1 once assembled.
        """
        post = d.site_xpos[self.frame_site] - d.site_xpos[self.frame_mount_site]
        upright = jnp.clip(post[2] / jnp.linalg.norm(post), 0., 1.)
        tip = d.site_xpos[self.frame_tip_site]
        slot = jnp.mean(d.geom_xpos[np.asarray(self.stand_walls)], axis=0)
        offset = jnp.linalg.norm(tip[:2] - slot[:2])
        place = d.geom_xpos[self.base_geom, 2] + 0.005
        clearance = d.site_xpos[self.stand_mount_site, 2] + 0.02
        carry = _lift_and_carry(tip[2], offset, place, clearance, self.frame_carry_scale)
        held = self.grasp.held(d, np.array([True, False]))
        released = ~held & (offset < 0.01) & (upright > 0.95)
        near = _closing(jnp.linalg.norm(self.arm.eef_pos(d) - d.geom_xpos[self.frame_grip]), _REACH_SCALE)
        near = near * self._grip_alignment(d, self.frame_grip, 1)
        reach = jnp.where(released, 1., near)
        grip = _grip(self.arm, self.grasp, d, held | released, d.geom_xpos[self.frame_grip], np.array([True, False]))
        turn = self._frame_turn_error(d)
        aligned = _fine(turn, np.pi, _FRAME_YAW_FINE)
        up, over, down = carry
        carry = (up * upright, over * upright * (0.5 + 0.5 * aligned), down * aligned)
        centred = aligned * _fine(offset, _ALIGN, _SLOT_FINE)
        inserted = centred * _ramp(jnp.maximum(tip[2] - place, 0.), clearance - place)
        progress = _progress(reach, grip, held | released, upright, _ramp(turn, np.pi), aligned, *carry, centred, inserted)
        return jnp.where(self._frame_assembled(d), 1., progress * jnp.where(released, 1., self._joint_room(d)))

    def _tool_progress(self, d):
        """
        Reaching the tool's grip, holding the tool, bringing its hole onto the hook just past the hook's end, and
        centering the hole on the hook to within its tolerance; held and reached once released there, and 1 once the tool
        hangs on the frame.
        """
        hang = d.site_xpos[self.frame_hang_site]
        hook = d.site_xpos[self.frame_site] - hang
        hook_length = jnp.linalg.norm(hook)
        hook = hook / hook_length
        hole = d.site_xpos[self.tool_hole_center]
        along = jnp.dot(hole - hang, hook)
        off_hook = jnp.linalg.norm(hole - hang - along * hook)
        threaded = _approach(off_hook, jnp.maximum(0.06 * hook_length - along, 0.), self.tool_carry_scale, hook_length)
        centred = _fine(off_hook, _FINE_ALIGN, self.hole_tolerance)
        held = self.grasp.held(d, np.array([False, True]))
        released = ~held & (jnp.linalg.norm(hole - hang) < 0.02)
        near = _closing(jnp.linalg.norm(self.arm.eef_pos(d) - d.geom_xpos[self.tool_grip]), _REACH_SCALE)
        near = near * self._grip_alignment(d, self.tool_grip, -2)
        reach = jnp.where(released, 1., near)
        grip = _grip(self.arm, self.grasp, d, held | released, d.geom_xpos[self.tool_grip], np.array([False, True]))
        return jnp.where(self._tool_on_frame(d), 1., _progress(reach, grip, held | released, threaded, centred))

    def progress(self, d):
        """Half for assembling the frame and half for hanging the tool on it once assembled."""
        return 0.5 * self._frame_progress(d) + 0.5 * jnp.where(self._frame_assembled(d), self._tool_progress(d), 0.)


class TwoArmTransport(Task):
    def __init__(self, spec, m, arms):
        self.arms = arms
        self.payload, self.trash = m.body(spec["payload"]).id, m.body(spec["trash"]).id
        self.lid_handle = m.geom(spec["lid_handle"]).id
        self.payload_handle = m.geom(spec["payload_handle"]).id
        self.handle_reach = float(m.geom_size[self.payload_handle, 2]) - _HANDLE_END
        self.target_bin, self.trash_bin = m.geom(spec["target_bin"]).id, m.geom(spec["trash_bin"]).id
        self.payload_in_target_bin = ContactTest(m, _ids(m.geom, spec["target_bin_geoms"]), _ids(m.geom, spec["payload_geoms"]))
        self.trash_in_trash_bin = ContactTest(m, _ids(m.geom, spec["trash_bin_geoms"]), _ids(m.geom, spec["trash_geoms"]))
        self.reward_scale = spec["reward_scale"]
        self.start_bin = m.geom(spec["start_bin"]).id
        self.lid = m.geom(spec["lid_geoms"][0]).id
        self.bin_half = float(m.geom_size[self.start_bin][0])
        objects = [spec["trash_geoms"], spec["lid_geoms"], spec["payload_geoms"]]
        self.grasps = [_Grasp(m, pads, objects) for pads in spec["pad_geoms"]]
        self.trash_carry_scale, self.payload_carry_scale = spec["trash_carry_scale"], spec["payload_carry_scale"]

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

    def _held_by(self, d, i):
        """Whether each gripper holds object `i`: 0 the trash, 1 the lid, 2 the payload."""
        mask = np.arange(3) == i
        return jnp.stack([grasp.held(d, mask) for grasp in self.grasps])

    def _held(self, d, i, arms=(0, 1)):
        """Whether one of grippers `arms` holds object `i`."""
        return jnp.any(self._held_by(d, i)[np.asarray(arms)])

    def _reach(self, d, target, arms=(0, 1), grip=None):
        """
        Closeness of the nearest of grippers `arms` approaching `target` from above, or, given the payload's handle
        `grip`, the point of its long axis nearest the gripper, counting from half to all as its fingers close across
        the handle (`_across_handle`).
        """
        def closeness(a):
            if grip is None:
                return _reach(self.arms[a], d, target)
            axis = d.geom_xmat[grip].reshape(3, 3)[:, 2]
            centre = d.geom_xpos[grip]
            along = jnp.clip((self.arms[a].eef_pos(d) - centre) @ axis, -self.handle_reach, self.handle_reach)
            return _reach(self.arms[a], d, centre + along * axis) * (0.5 + 0.5 * self._across_handle(d, a))
        return jnp.max(jnp.stack([closeness(a) for a in arms]))

    def _across_handle(self, d, a):
        """
        1 while gripper `a`'s fingers close across the payload's handle within `_ACROSS_SLACK` of its long axis, falling
        to 0 over a further `_ACROSS`.
        """
        g = self.grasps[a]
        fingers = d.geom_xpos[g.pads[1]] - d.geom_xpos[g.pads[0]]
        fingers = fingers / jnp.maximum(jnp.linalg.norm(fingers), 1e-6)
        along = jnp.abs(fingers @ d.geom_xmat[self.payload_handle].reshape(3, 3)[:, 2])
        return _ramp(jnp.maximum(along - _ACROSS_SLACK, 0.), _ACROSS)

    def _toward(self, d, a, target, grip=None):
        """
        Closeness of gripper `a` to `target`, or, given the payload's handle `grip`, to the point of its long axis
        nearest the gripper, falling linearly over `_HANDOVER_REACH`, and counting from half to all as its fingers
        close across the handle.
        """
        eef = self.arms[a].eef_pos(d)
        if grip is None:
            return _ramp(jnp.linalg.norm(eef - target), _HANDOVER_REACH)
        axis = d.geom_xmat[grip].reshape(3, 3)[:, 2]
        centre = d.geom_xpos[grip]
        point = centre + jnp.clip((eef - centre) @ axis, -self.handle_reach, self.handle_reach) * axis
        return _ramp(jnp.linalg.norm(eef - point), _HANDOVER_REACH) * (0.5 + 0.5 * self._across_handle(d, a))

    def _grip(self, d, i, held, target, arms=(0, 1)):
        """`_grip` of the best of grippers `arms` on object `i` at grip point `target`, 1 while `held`."""
        mask = np.arange(3) == i
        return jnp.max(jnp.stack([_grip(self.arms[a], self.grasps[a], d, held, target, mask) for a in arms]))

    def _upright(self, d, arms):
        """How upright the more upright of grippers `arms` points."""
        return jnp.max(jnp.stack([_upright(self.arms[a], d) for a in arms]))

    def _into_bin(self, d, i, pos, bin_base, placed, scale, arms, handover=None):
        """
        Progress of object `i` at `pos` into the bin of `bin_base`: reaching it from above, holding it, lifting it over
        the bin's walls, carrying it over the bin, and lowering it in, the payload's holding and carrying counting as far
        as its holder's fingers close across its handle and its gripper is commanded closed, and, given arm `handover`,
        that arm closing in on it (`_toward`) and gripping it until it holds it; held and reached once released over the bin within
        `_RELEASE_HEIGHT` of its floor, and 1 once `placed`.
        """
        center = d.geom_xpos[bin_base]
        inner = self.bin_half - 0.03
        offset = jnp.linalg.norm(pos[:2] - jnp.clip(pos[:2], center[:2] - inner, center[:2] + inner))
        carry = _lift_and_carry(pos[2], offset, center[2] + 0.03, center[2] + 0.2, scale)
        held = self._held(d, i, arms)
        height = pos[2] - (center[2] + 0.03)
        released = (offset == 0.) & ~held & (height > -_ALIGN) & (height < _RELEASE_HEIGHT)
        reach = jnp.where(released, 1., self._reach(d, pos, arms, self.payload_handle if i == 2 else None))
        grip = self._grip(d, i, held | released, pos, arms)
        if i == 2:
            by = self._held_by(d, i)
            holders = jnp.stack([
                jnp.where(by[a], self._across_handle(d, a) * self.arms[a].grip_closure(d), 0.) for a in arms
            ])
            firm = jnp.where(released | ~held, 1., jnp.max(holders))
            grip = grip * firm
            carry = tuple(stage * firm for stage in carry)
        if handover is not None:
            taken = self._held(d, i, (handover,)) | released
            handle = self.payload_handle if i == 2 else None
            closing = 0.5 * self._toward(d, handover, pos, handle) + 0.5 * self._grip(d, i, taken, pos, (handover,))
            carry = (*carry, jnp.where(taken, 1., closing))
        return jnp.where(placed, 1., _progress(reach, grip, held | released, *carry))

    def _lid_progress(self, d):
        """
        Reaching the lid's handle, holding the lid, and sliding it off the start bin, until at most 6 cm of it still
        covers the bin; held and reached once released more than halfway off, and 1 once off.
        """
        moved = jnp.linalg.norm(d.geom_xpos[self.lid, :2] - d.geom_xpos[self.start_bin, :2]) / (2. * self.bin_half - 0.06)
        held = self._held(d, 1, (0,))
        released = ~held & (moved > 0.5)
        reach = jnp.where(released, 1., self._reach(d, d.geom_xpos[self.lid_handle], (0,)))
        grip = self._grip(d, 1, held | released, d.geom_xpos[self.lid_handle], (0,))
        return jnp.where(moved >= 1., 1., _progress(reach, grip, held | released, jnp.clip(moved, 0., 1.)))

    def progress(self, d):
        """
        Half for the trash arm (robot 1) moving the trash into the trash bin, and half for the payload arm (robot 0):
        0.3 of it for taking the lid off the start bin and 0.7 for moving the payload into the target bin with either
        arm, handing it to the trash arm, once the lid is off; each scaled from 0.5 to 1 by how upright the gripper doing it points.
        """
        trash = self._into_bin(
            d, 0, d.xpos[self.trash], self.trash_bin, self.trash_in_trash_bin(d), self.trash_carry_scale, (1,),
        ) * _tilt(self._upright(d, (1,)))
        lid = self._lid_progress(d)
        payload = self._into_bin(
            d, 2, d.xpos[self.payload], self.target_bin, self.payload_in_target_bin(d), self.payload_carry_scale, (0, 1),
            handover=1,
        )
        held_by = self._held_by(d, 2)
        payload_arms = jnp.where(held_by.any(), jnp.where(held_by[1], 1, 0), -1)
        payload_upright = jnp.where(payload_arms < 0, self._upright(d, (0, 1)), jnp.stack([_upright(a, d) for a in self.arms])[payload_arms])
        payload_arm = 0.3 * lid * _tilt(_upright(self.arms[0], d)) + 0.7 * jnp.where(lid == 1., payload * _tilt(payload_upright), 0.)
        return 0.5 * trash + 0.5 * payload_arm


def _pose_to_gripper(arm, d, body):
    """MimicGen's `{obj}_pos`, `_quat`, `_to_robot0_eef_pos` and `_to_robot0_eef_quat` of `body`, from the current pose."""
    pos, quat = _body_pose_xyzw(d, body)
    rel_pos, rel_quat = _relative_to_gripper(pos, quat, arm, d)
    return [pos, quat, rel_pos, rel_quat]


def _gripper_in(arm, d, body):
    """Pose of the grip site in the frame of `body`: MimicGen's `robot0_eef_pos_rel_*` and `_quat_rel_*`."""
    return relative_pose(d.site_xpos[arm.site], d.site_xmat[arm.site].reshape(3, 3), d.xpos[body], quat2mat(wxyz_to_xyzw(d.xquat[body])))


class Stack(Task):
    """MimicGen's Stack: cube A lifted onto cube B and let go."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.cube_a, self.cube_b = m.body(spec["cube_a"]).id, m.body(spec["cube_b"]).id
        self.table_height = spec["table_height"]
        self.half_a, self.half_b = spec["cube_half_sizes"]
        self.carry_scale = spec["carry_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["cube_a_geoms"]])
        self.touching = ContactTest(m, _ids(m.geom, spec["cube_a_geoms"]), _ids(m.geom, spec["cube_b_geoms"]))

    def object_obs(self, d, cache):
        (pa, qa), (pb, qb) = _body_pose_xyzw(d, self.cube_a), _body_pose_xyzw(d, self.cube_b)
        eef = self.arm.eef_pos(d)
        return jnp.concatenate([pa, qa, pb, qb, pa - eef, pb - eef, pb - pa])

    def _lifted(self, d):
        return d.xpos[self.cube_a, 2] > self.table_height + 0.04

    def success(self, d):
        return ~self.grasp(d) & self._lifted(d) & self.touching(d)

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def shaped_reward(self, d):
        """robosuite's staged reward: the largest of reaching and grasping cube A, lifting it over B, and stacking it, halved."""
        pa, pb = d.xpos[self.cube_a], d.xpos[self.cube_b]
        grasping = self.grasp(d)
        reach = 0.25 * _closeness(jnp.linalg.norm(self.arm.eef_pos(d) - pa)) + 0.25 * grasping
        lift = jnp.where(self._lifted(d), 1. + 0.5 * (1. - jnp.tanh(jnp.linalg.norm(pa[:2] - pb[:2]))), 0.)
        stack = jnp.where(self.success(d), 2., 0.)
        return jnp.maximum(jnp.maximum(reach, lift), stack) / 2.

    def progress(self, d):
        """
        Reaching cube A, holding it, lifting it over cube B and lowering it onto B; held once released over B, resting
        on it or dropping onto it from within a cube's height.
        """
        progress, _ = _stack_progress(
            self.arm, self.grasp, d, self.cube_a, self.cube_b, self.half_a, self.half_b, self.touching(d), self._lifted(d),
            self.carry_scale,
        )
        return progress


def _stack_progress(arm, grasp, d, cube, base, half, half_base, touching, lifted, carry_scale):
    """
    Progress of stacking `cube` (half size `half`, grasped by `grasp`) on `base` (half size `half_base`): reaching it,
    holding it, lifting it over `base` and lowering it on top, held once released over `base`, `touching` it or
    dropping onto it from within a cube's height, while `lifted`. Also returns whether it is so released.
    """
    p, p_base = d.xpos[cube], d.xpos[base]
    held = grasp.held(d)
    place = p_base[2] + half_base + half
    offset = jnp.linalg.norm(p[:2] - p_base[:2])
    dropping = (p[2] > place) & (p[2] < place + 2. * half)
    released = ~held & (touching | dropping) & (offset < half_base) & lifted
    reach = jnp.where(released, 1., _reach(arm, d, p))
    grip = _grip(arm, grasp, d, held | released, p)
    carry = _lift_and_carry(p[2], offset, place, place + 2. * half, carry_scale)
    return _progress(reach, grip, held | released, *carry), released


class StackThree(Task):
    """MimicGen's StackThree: cube A stacked on cube B, then cube C on cube A, both let go."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.cube_a, self.cube_b, self.cube_c = (m.body(spec[k]).id for k in ("cube_a", "cube_b", "cube_c"))
        self.table_height = spec["table_height"]
        self.half_a, self.half_b, self.half_c = spec["cube_half_sizes"]
        self.lift_a, self.lift_c = spec["lift_margins"]
        self.carry_scale = spec["carry_scale"]
        self.grasp_a = _Grasp(m, spec["pad_geoms"], [spec["cube_a_geoms"]])
        self.grasp_c = _Grasp(m, spec["pad_geoms"], [spec["cube_c_geoms"]])
        a, b, c = (_ids(m.geom, spec[k]) for k in ("cube_a_geoms", "cube_b_geoms", "cube_c_geoms"))
        self.a_on_b = ContactTest(m, a, b)
        self.c_on_a = ContactTest(m, c, a)

    def object_obs(self, d, cache):
        """Stack's 23 entries, then cube C's pose, its position relative to the gripper and to A, and B's relative to C."""
        (pa, qa), (pb, qb), (pc, qc) = (_body_pose_xyzw(d, body) for body in (self.cube_a, self.cube_b, self.cube_c))
        eef = self.arm.eef_pos(d)
        return jnp.concatenate([pa, qa, pb, qb, pa - eef, pb - eef, pb - pa, pc, qc, pc - eef, pc - pa, pb - pc])

    def _a_lifted(self, d):
        return d.xpos[self.cube_a, 2] > self.table_height + self.lift_a

    def _c_lifted(self, d):
        return d.xpos[self.cube_c, 2] > self.table_height + self.lift_c

    def success(self, d):
        a_stacked = ~self.grasp_a(d) & self._a_lifted(d) & self.a_on_b(d)
        c_stacked = ~self.grasp_c(d) & self._c_lifted(d) & self.c_on_a(d)
        return a_stacked & c_stacked

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def progress(self, d):
        """Half for stacking cube A on cube B; half for stacking cube C on cube A, once A is released on B."""
        a, a_done = _stack_progress(
            self.arm, self.grasp_a, d, self.cube_a, self.cube_b, self.half_a, self.half_b, self.a_on_b(d),
            self._a_lifted(d), self.carry_scale,
        )
        c, _ = _stack_progress(
            self.arm, self.grasp_c, d, self.cube_c, self.cube_a, self.half_c, self.half_a, self.c_on_a(d),
            self._c_lifted(d), self.carry_scale,
        )
        return 0.5 * a + 0.5 * jnp.where(a_done, c, 0.)


class Threading(Task):
    """MimicGen's Threading: the needle's bar threaded through the tripod's ring."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.needle, self.tripod = m.body(spec["needle"]).id, m.body(spec["tripod"]).id
        self.bar, self.handle = m.geom(spec["bar"]).id, m.geom(spec["handle"]).id
        self.rings = np.asarray(_ids(m.geom, spec["ring_geoms"]))
        self.ring_radius = spec["ring_radius"]
        self.carry_scale = spec["carry_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["needle_geoms"]])

    def object_obs(self, d, cache):
        return jnp.concatenate([*_pose_to_gripper(self.arm, d, self.needle), *_pose_to_gripper(self.arm, d, self.tripod)])

    def _bar_to_ring(self, d):
        return jnp.linalg.norm(d.geom_xpos[self.bar] - jnp.mean(d.geom_xpos[self.rings], axis=0))

    def success(self, d):
        return self._bar_to_ring(d) < self.ring_radius

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def progress(self, d):
        """Reaching the needle's handle, holding the needle, and bringing its bar's center to the ring's center."""
        held = self.grasp.held(d)
        handle = d.geom_xpos[self.handle]
        reach = _reach(self.arm, d, handle)
        grip = _grip(self.arm, self.grasp, d, held, handle)
        dist = self._bar_to_ring(d)
        return _progress(reach, grip, held, _closing(dist, self.carry_scale), _fine(dist, self.carry_scale, self.ring_radius))


class Coffee(Task):
    """MimicGen's Coffee: the pod put into the coffee machine's holder and the lid closed over it."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.pod, self.machine = m.body(spec["pod"]).id, m.body(spec["machine"]).id
        self.holder, self.lid = m.body(spec["holder"]).id, m.body(spec["lid"]).id
        self.hinge = int(m.jnt_qposadr[m.joint(spec["hinge"]).id])
        self.hinge_open = float(m.jnt_range[m.joint(spec["hinge"]).id, 1])
        self.pod_half, self.holder_half = spec["pod_half_size"], spec["holder_half_size"]
        self.lid_half_height = spec["lid_half_height"]
        self.pod_rest_height = spec["pod_rest_height"]
        self.lid_closed = spec["lid_closed"]
        self.carry_scale = spec["carry_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["pod_geoms"]])

    def object_obs(self, d, cache):
        poses = [part for body in (self.pod, self.machine, self.holder, self.lid) for part in _pose_to_gripper(self.arm, d, body)]
        return jnp.concatenate([*poses, d.qpos[self.hinge][None]])

    def extra_obs(self, d):
        """MimicGen's grip-site poses relative to the pod and to the pod holder."""
        (pod_pos, pod_quat), (holder_pos, holder_quat) = _gripper_in(self.arm, d, self.pod), _gripper_in(self.arm, d, self.holder)
        prefix = self.arm.prefix
        return {
            f"{prefix}eef_pos_rel_pod": pod_pos, f"{prefix}eef_quat_rel_pod": pod_quat,
            f"{prefix}eef_pos_rel_pod_holder": holder_pos, f"{prefix}eef_quat_rel_pod_holder": holder_quat,
        }

    def _pod_in_holder(self, d):
        pod, holder, lid = d.xpos[self.pod], d.xpos[self.holder], d.xpos[self.lid]
        centered = jnp.linalg.norm(pod[:2] - holder[:2]) <= self.holder_half[0] - self.pod_half[0]
        above_floor = pod[2] - self.pod_half[2] >= holder[2] - self.holder_half[2]
        below_lid = pod[2] + self.pod_half[2] <= lid[2] - self.lid_half_height
        return centered & above_floor & below_lid

    def success(self, d):
        return (d.qpos[self.hinge] < self.lid_closed) & self._pod_in_holder(d)

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def progress(self, d):
        """
        Half for the pod, reaching it, holding it, lifting it over the holder and lowering it in, held once released in
        the holder; half for closing the lid, once the pod is in the holder.
        """
        pod, holder = d.xpos[self.pod], d.xpos[self.holder]
        held = self.grasp.held(d)
        inside = self._pod_in_holder(d)
        released = ~held & inside
        place = holder[2] + self.pod_rest_height
        offset = jnp.linalg.norm(pod[:2] - holder[:2])
        reach = jnp.where(released, 1., _reach(self.arm, d, pod))
        grip = _grip(self.arm, self.grasp, d, held | released, pod)
        carry = _lift_and_carry(pod[2], offset, place, holder[2] + self.holder_half[2] + 2. * self.pod_half[2], self.carry_scale)
        pod_progress = _progress(reach, grip, held | released, *carry)
        closed = _ramp(jnp.maximum(d.qpos[self.hinge] - self.lid_closed, 0.), self.hinge_open - self.lid_closed)
        return 0.5 * pod_progress + 0.5 * jnp.where(released, closed, 0.)


# height above its resting height within which a mug let go over the drawer's bottom counts as released into it
_MUG_DROP = 0.03


class MugCleanup(Task):
    """MimicGen's MugCleanup: the drawer opened, the mug put upright into it, and the drawer closed."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.mug, self.drawer = m.body(spec["mug"]).id, m.body(spec["drawer"]).id
        self.slide = int(m.jnt_qposadr[m.joint(spec["slide"]).id])
        self.bottom, self.handle = m.geom(spec["bottom"]).id, m.geom(spec["handle"]).id
        self.closed, self.opened = spec["drawer_closed"], spec["drawer_open"]
        self.upright_tol, self.release_tilt = spec["upright_tol"], spec["release_tilt"]
        self.place_height, self.clearance = spec["place_height"], spec["clearance"]
        self.carry_scale = spec["carry_scale"]
        self.bottom_half = m.geom_size[self.bottom, :2].copy()
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["mug_geoms"]])
        self.on_bottom = ContactTest(m, [self.bottom], _ids(m.geom, spec["mug_geoms"]))

    def object_obs(self, d, cache):
        """The mug's and the drawer's poses and poses relative to the gripper, then the drawer's slide position."""
        return jnp.concatenate([
            *_pose_to_gripper(self.arm, d, self.mug), *_pose_to_gripper(self.arm, d, self.drawer), d.qpos[self.slide][None],
        ])

    def _upright(self, d, tol):
        return d.xmat[self.mug].reshape(3, 3)[2, 2] > 1. - tol

    def _over_bottom(self, d):
        """Whether the mug stands within `release_tilt` of upright over the drawer's bottom, at most `_MUG_DROP` above resting."""
        rel = d.geom_xmat[self.bottom].reshape(3, 3).T @ (d.xpos[self.mug] - d.geom_xpos[self.bottom])
        return self._upright(d, self.release_tilt) & jnp.all(jnp.abs(rel[:2]) < self.bottom_half) & (rel[2] < self.place_height + _MUG_DROP)

    def success(self, d):
        return (d.qpos[self.slide] > self.closed) & self._upright(d, self.upright_tol) & self.on_bottom(d)

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def _handle_stage(self, d, done):
        """Closeness of the gripper to the drawer's handle, superseded by `done` in [0, 1]."""
        reach = _closing(jnp.linalg.norm(self.arm.eef_pos(d) - d.geom_xpos[self.handle]), _REACH_SCALE)
        return 0.3 * jnp.maximum(reach, done) + 0.7 * done

    def progress(self, d):
        """
        A quarter for pulling the drawer open by its handle; half for the mug, reaching it, holding it, lifting it over
        the open drawer and lowering it onto the drawer's bottom, held once released upright over the bottom; a quarter
        for pushing the drawer closed, once the mug is in it.
        """
        q = d.qpos[self.slide]
        mug, bottom = d.xpos[self.mug], d.geom_xpos[self.bottom]
        held = self.grasp.held(d)
        released = ~held & self._over_bottom(d)
        opened = _ramp(jnp.maximum(q - self.opened, 0.), -self.opened)
        closed = _ramp(jnp.maximum(self.closed - q, 0.), self.closed - self.opened)
        offset = jnp.linalg.norm(mug[:2] - bottom[:2])
        reach = jnp.where(released, 1., _reach(self.arm, d, mug))
        grip = _grip(self.arm, self.grasp, d, held | released, mug)
        carry = _lift_and_carry(mug[2], offset, bottom[2] + self.place_height, bottom[2] + self.clearance, self.carry_scale)
        mug_progress = _progress(reach, grip, (held | released) * jnp.where(released, 1., opened), *carry)
        open_progress = jnp.where(released, 1., self._handle_stage(d, opened))
        close_progress = jnp.where(released, self._handle_stage(d, closed), 0.)
        return 0.25 * open_progress + 0.5 * mug_progress + 0.25 * close_progress


class _MimicGenPlacement:
    """
    robosuite 1.4.1's placement observations: per observed object its current position, quaternion, and pose relative
    to the gripper, then the positions of `observed_bodies`. Without `single`, progress covers every object.
    """
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.observed_bodies = _ids(m.body, spec.get("observed_bodies", []))

    def cache(self, d):
        return ()

    def object_obs(self, d, cache):
        parts = [part for i in self.observed for part in _pose_to_gripper(self.arm, d, self.bodies[i])]
        return jnp.concatenate(parts + [d.xpos[b] for b in self.observed_bodies])

    def _object_progress(self, d, i):
        """
        Progress of object `i` without the gripper's uprightness, and whether it is released over its target: as
        `_PlacementTask.progress`.
        """
        pos = d.xpos[self.bodies[i]]
        mask = np.arange(len(self.bodies)) == i
        grasped = self.grasp.held(d, mask)
        offset = self._target_offset(d, i)
        carry = self._carry(d, i, pos, offset)
        released = self._released(d, i, pos, offset, grasped)
        target = self._reach_targets(d)[i]
        reach = jnp.where(released, 1., _reach(self.arm, d, target))
        grip = _grip(self.arm, self.grasp, d, grasped | released, target, mask)
        let_go = jnp.where(released, jnp.clip(jnp.linalg.norm(self.arm.eef_pos(d) - pos) / _LET_GO, 0., 1.), 0.)
        return _progress(reach, grip, grasped | released, *carry, let_go), released

    def _released(self, d, i, pos, offset, grasped):
        """Whether object `i` at `pos`, at horizontal distance `offset` from its target, is let go of where it is placed."""
        height = pos[2] - self._place_height(d)
        return (offset == 0.) & ~grasped & (height > -_ALIGN) & (height < _RELEASE_HEIGHT) & self._releasable(d, i, pos)

    def progress(self, d):
        """
        With `single`, the progress of the observed object, with the gripper upright. Otherwise the mean over the objects
        of: each released object's progress, and, for the one unreleased object furthest along, its progress with the
        gripper upright; the others count 0.
        """
        upright = _upright(self.arm, d)
        if self.single:
            return self._object_progress(d, self.observed[0])[0] * upright
        each = [self._object_progress(d, i) for i in range(len(self.bodies))]
        progress = jnp.stack([p for p, _ in each])
        released = jnp.stack([r for _, r in each])
        done = jnp.sum(jnp.where(released, progress, 0.))
        best_open = jnp.max(jnp.where(released, 0., progress))
        return (done + best_open * upright) / len(self.bodies)


class MimicGenNutAssembly(_MimicGenPlacement, NutAssembly):
    """MimicGen's Square_D0/D1/D2 and NutAssembly_D0: nuts onto their pegs."""


class MimicGenPickPlace(_MimicGenPlacement, PickPlace):
    """MimicGen's PickPlace_D0: every object into its bin, each lowered to its own resting height."""
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.place_heights = np.asarray(spec["place_heights"])

    def _carry(self, d, i, pos, offset):
        return _lift_and_carry(pos[2], offset, self.place_heights[i], self._clearance(d), self.carry_scale)

    def _released(self, d, i, pos, offset, grasped):
        """Whether object `i` is let go of over its bin, below `_RELEASE_HEIGHT` above its resting height, standing or not."""
        over = jnp.all((self.bin_low[i] < pos[:2]) & (pos[:2] < self.bin_high[i]))
        return over & ~grasped & (self.bin_z < pos[2]) & (pos[2] < self.place_heights[i] + _RELEASE_HEIGHT)


# distance of the gripper from a handle or button pad within which reaching it counts finely
_HANDLE_FINE = 0.05
# distance of the gripper from the pot within which reaching it to push it counts finely
_PUSH_REACH = 0.1


# force on the gripper above which robosuite-task-zoo's environments report contact
_CONTACT_FORCE = 2.


def _force_obs(arm, sensor, d):
    """robosuite-task-zoo's observations of the force on `arm`'s gripper, measured by force sensor `sensor`."""
    norm = jnp.linalg.norm(d.sensordata[sensor + np.arange(3)])
    return {f"{arm.prefix}eef_force_norm": norm[None], f"{arm.prefix}contact": (norm > _CONTACT_FORCE)[None]}


class HammerCleanup(Task):
    """MimicGen's HammerCleanup: the hammer put into the drawer and the drawer closed."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.hammer, self.cabinet = m.body(spec["hammer"]).id, m.body(spec["cabinet"]).id
        self.slide = int(m.jnt_qposadr[m.joint(spec["slide"]).id])
        self.bottom, self.handle = m.geom(spec["drawer_bottom"]).id, m.geom(spec["drawer_handle"]).id
        self.closed, self.opened, self.ajar = spec["drawer_closed"], spec["drawer_opened"], spec["drawer_ajar"]
        self.bottom_half = np.asarray(m.geom_size[self.bottom])
        self.in_drawer_z, self.in_drawer_y = spec["in_drawer_z"], spec["in_drawer_y"]
        self.target_offset = np.asarray(spec["drawer_target_offset"])
        self.clearance, self.release_height = spec["drawer_clearance"], spec["release_height"]
        self.carry_scale = spec["carry_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["hammer_geoms"]])
        self.force = int(m.sensor_adr[m.sensor(spec["force_sensor"]).id])
        self.on_bottom = ContactTest(m, [self.bottom], _ids(m.geom, spec["hammer_geoms"]))

    def object_obs(self, d, cache):
        bodies = (self.hammer, self.cabinet)
        return jnp.concatenate([part for body in bodies for part in _pose_to_gripper(self.arm, d, body)])

    def extra_obs(self, d):
        return _force_obs(self.arm, self.force, d)

    def _in_drawer(self, d):
        """The success test of the hammer's placement: within a world box (D0) or touching the drawer's bottom (D1)."""
        if self.in_drawer_z is None:
            return self.on_bottom(d)
        pos = d.xpos[self.hammer]
        return (pos[2] > self.in_drawer_z[0]) & (pos[2] < self.in_drawer_z[1]) & (pos[1] > self.in_drawer_y)

    def success(self, d):
        return self._in_drawer(d) & (d.qpos[self.slide] > self.closed)

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def _target(self, d):
        """Where the hammer rests on the part of the drawer's bottom outside the cabinet when the drawer is open."""
        return d.geom_xpos[self.bottom] + d.geom_xmat[self.bottom].reshape(3, 3) @ self.target_offset

    def _over_bottom(self, d):
        """Whether the hammer is above the drawer's bottom, within `release_height` of it."""
        rel = d.geom_xmat[self.bottom].reshape(3, 3).T @ (d.xpos[self.hammer] - d.geom_xpos[self.bottom])
        return jnp.all(jnp.abs(rel[:2]) < self.bottom_half[:2]) & (rel[2] > 0.) & (rel[2] < self.release_height)

    def progress(self, d):
        """
        A third each for pulling the drawer open by its handle; reaching, holding and carrying the hammer onto the
        drawer's bottom once the drawer is ajar, held once released over it; and pushing the drawer closed.
        """
        q = d.qpos[self.slide]
        eef = self.arm.eef_pos(d)
        handle = _fine(jnp.linalg.norm(eef - d.geom_xpos[self.handle]), _REACH_SCALE, _HANDLE_FINE)
        opening = 0.5 * handle + 0.5 * _ramp(jnp.maximum(q - self.opened, 0.), -self.opened)
        closing = 0.5 * handle + 0.5 * _ramp(jnp.maximum(self.closed - q, 0.), self.closed - self.opened)

        hammer = d.xpos[self.hammer]
        target = self._target(d)
        held = self.grasp.held(d)
        placed = self.on_bottom(d) | self._in_drawer(d) | self._over_bottom(d)
        released = ~held & placed
        reach = jnp.where(released, 1., _reach(self.arm, d, hammer))
        grip = _grip(self.arm, self.grasp, d, held | released, hammer)
        offset = jnp.linalg.norm(hammer[:2] - target[:2])
        carry = _lift_and_carry(hammer[2], offset, target[2], target[2] + self.clearance, self.carry_scale)
        carrying = _progress(reach, grip, held | released, *carry)
        return jnp.where(released, (2. + closing) / 3., jnp.where(q < self.ajar, (1. + carrying) / 3., opening / 3.))


class Kitchen(Task):
    """
    MimicGen's Kitchen: the stove turned on, the pot put on it, the bread put in the pot, the pot pushed into the
    serving region, and the stove turned off. Success depends on whether the stove has ever been on with the pot on it
    and the bread in the pot, a flag kept outside the simulator state in the task's memory.
    """
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.bodies = _ids(m.body, spec["object_bodies"])
        self.stove, self.bread, self.pot = m.body(spec["stove"]).id, m.body(spec["bread"]).id, m.body(spec["pot"]).id
        self.region = m.body(spec["serving_region"]).id
        self.hinge = int(m.jnt_qposadr[m.joint(spec["button_hinge"]).id])
        self.pad = m.geom(spec["button_pad"]).id
        self.button_off, self.button_on = spec["button_off"], spec["button_on"]
        self.button_max = float(m.jnt_range[m.joint(spec["button_hinge"]).id, 1])
        self.region_half = jnp.asarray(spec["region_half_size"])
        self.pot_walls = np.asarray(_ids(m.geom, spec["pot_grip_geoms"]))
        self.pot_on_stove, self.pot_on_region = spec["pot_on_stove_height"], spec["pot_on_region_height"]
        self.bread_in_pot = spec["bread_in_pot_height"]
        self.pot_clearance, self.bread_clearance = spec["pot_clearance"], spec["bread_clearance"]
        self.pot_inner_half = np.asarray(spec["pot_inner_half_size"])
        self.release_height = spec["bread_release_height"]
        self.stove_half, self.pot_release_height = spec["stove_half_size"], spec["pot_release_height"]
        self.carry_scale = spec["carry_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["pot_geoms"], spec["bread_geoms"]])
        self.bread_touches_pot = ContactTest(m, _ids(m.geom, spec["bread_geoms"]), _ids(m.geom, spec["pot_geoms"]))
        self.pot_on_burner = ContactTest(m, [m.geom(spec["pot_bottom"]).id], [m.geom(spec["burner"]).id])
        self.force = int(m.sensor_adr[m.sensor(spec["force_sensor"]).id])

    def object_obs(self, d, cache):
        return jnp.concatenate([part for body in self.bodies for part in _pose_to_gripper(self.arm, d, body)])

    def extra_obs(self, d):
        return _force_obs(self.arm, self.force, d)

    def _on(self, d):
        return d.qpos[self.hinge] >= self.button_on

    def _in_region(self, d):
        return jnp.all(jnp.abs(d.xpos[self.region] - d.xpos[self.pot]) < self.region_half)

    def _cooking(self, d):
        return self._on(d) & self.bread_touches_pot(d) & self.pot_on_burner(d)

    def initial_memory(self):
        return (jnp.zeros((), bool),)

    def remember(self, d, memory):
        """Whether the stove has been on with the pot on it and the bread in the pot, in `d` or before."""
        cooked, = memory
        return (cooked | self._cooking(d),)

    def success(self, d, cooked):
        return self._in_region(d) & ~self._on(d) & self.bread_touches_pot(d) & cooked

    def reward(self, d, cooked):
        return self.success(d, cooked).astype(jnp.float32)

    def _bread_inside(self, d):
        """Whether the bread's center lies within the pot's walls, at most `release_height` above the pot."""
        pot_rot = quat2mat(wxyz_to_xyzw(d.xquat[self.pot]))
        rel = pot_rot.T @ (d.xpos[self.bread] - d.xpos[self.pot])
        return jnp.all(jnp.abs(rel[:2]) < self.pot_inner_half) & (rel[2] > 0.) & (rel[2] < self.release_height)

    def _pot_over_stove(self, d):
        """Whether the pot lies over the stove's top, at most `pot_release_height` above its resting height."""
        rel = d.xpos[self.pot] - d.xpos[self.stove]
        return jnp.all(jnp.abs(rel[:2]) < self.stove_half) & (rel[2] < self.pot_on_stove + self.pot_release_height)

    def _pot_grip(self, d):
        """The pot wall nearest the gripper, which the demonstrations pinch."""
        walls = d.geom_xpos[self.pot_walls]
        return walls[jnp.argmin(jnp.linalg.norm(walls - self.arm.eef_pos(d), axis=-1))]

    def _button(self, d, toward):
        """Reaching the button's pad and turning the switch from `button_off` to on (`toward` 1) or back off (-1)."""
        q = d.qpos[self.hinge]
        reach = _fine(jnp.linalg.norm(self.arm.eef_pos(d) - d.geom_xpos[self.pad]), _REACH_SCALE, _HANDLE_FINE)
        travel = self.button_on - self.button_off
        turned = jnp.where(
            toward > 0, _ramp(jnp.maximum(self.button_on - q, 0.), travel),
            _ramp(jnp.maximum(q - self.button_on, 0.), self.button_max - self.button_on),
        )
        return 0.5 * reach + 0.5 * turned

    def _carry(self, d, i, pos, target, place, clearance, placed):
        """Reaching, holding and carrying grasp object `i` (0 pot, 1 bread) at `pos` to `target`; held once `placed`."""
        objects = np.arange(2) == i
        held = self.grasp.held(d, objects)
        released = ~held & placed
        grip_point = self._pot_grip(d) if i == 0 else pos
        reach = jnp.where(released, 1., _reach(self.arm, d, grip_point))
        grip = _grip(self.arm, self.grasp, d, held | released, grip_point, objects)
        carry = _lift_and_carry(pos[2], jnp.linalg.norm(pos[:2] - target[:2]), place, clearance, self.carry_scale)
        return _progress(reach, grip, held | released, *carry)

    def progress(self, d, cooked):
        """
        A fifth for each stage in the demonstrations' order: turning the stove on, putting the pot on the stove, putting
        the bread in the pot, carrying or pushing the pot into the serving region with the bread inside it, and turning
        the stove off. The first three count as done once the stove has been on with the pot on it and the bread in the
        pot.
        """
        pot, bread, stove, region = d.xpos[self.pot], d.xpos[self.bread], d.xpos[self.stove], d.xpos[self.region]
        on, on_burner = self._on(d), self.pot_on_burner(d)
        place_stove = stove[2] + self.pot_on_stove
        pot_to_stove = self._carry(
            d, 0, pot, stove, place_stove, place_stove + self.pot_clearance, on_burner | self._pot_over_stove(d),
        )
        place_bread = pot[2] + self.bread_in_pot
        bread_to_pot = self._carry(d, 1, bread, pot, place_bread, pot[2] + self.bread_clearance, self._bread_inside(d))
        before = jnp.where(~on, self._button(d, 1), jnp.where(~on_burner, 1. + pot_to_stove, 2. + bread_to_pot))

        in_region, inside = self._in_region(d), self._bread_inside(d)
        place_region = region[2] + self.pot_on_region
        carried = self._carry(d, 0, pot, region, place_region, place_region + self.pot_clearance, in_region)
        pushed = 0.4 * _fine(jnp.linalg.norm(self.arm.eef_pos(d) - pot), _REACH_SCALE, _PUSH_REACH) + 0.6 * _fine(
            jnp.linalg.norm(pot[:2] - region[:2]), self.carry_scale, self.region_half[0],
        )
        pot_to_region = jnp.where(inside, jnp.maximum(carried, pushed), 0.)
        after = jnp.where(in_region & inside, 1. + self._button(d, -1), pot_to_region)
        return jnp.where(cooked, 3. + after, before) / 5.

def _yaw(d, body):
    R = d.xmat[body].reshape(3, 3)
    return jnp.arctan2(R[1, 0], R[0, 0])


class ThreePieceAssembly(Task):
    """MimicGen's ThreePieceAssembly: piece 1 dropped into the base's square hole, then piece 2 set over piece 1's ridge."""
    def __init__(self, spec, m, arms):
        self.arm = arms[0]
        self.base, self.piece_1, self.piece_2 = (m.body(spec[k]).id for k in ("base", "piece_1", "piece_2"))
        self.grip_1, self.grip_2 = m.geom(spec["piece_1_grip"]).id, m.geom(spec["piece_2_grip"]).id
        self.xy_thresh, self.z_thresh = spec["xy_thresh"], spec["z_thresh"]
        self.piece_2_height = spec["piece_2_height"]
        self.rest_1, self.rest_2 = spec["piece_1_rest_height"], spec["piece_2_rest_height"]
        self.clearance_1, self.clearance_2 = spec["piece_1_clearance"], spec["piece_2_clearance"]
        self.hole_half, self.slab_half = spec["hole_half_width"], spec["slab_half_width"]
        self.slot_half, self.ridge_half = spec["slot_half_width"], spec["ridge_half_size"]
        self.seat_slack, self.release_height = spec["seat_slack"], spec["release_height"]
        self.goal_tolerance, self.release_radius = spec["goal_tolerance"], spec["release_radius"]
        self.carry_scale = spec["carry_scale"]
        self.grasp = _Grasp(m, spec["pad_geoms"], [spec["piece_1_geoms"], spec["piece_2_geoms"]])
        self.only_1, self.only_2 = np.array([True, False]), np.array([False, True])

    def object_obs(self, d, cache):
        return jnp.concatenate([part for body in (self.base, self.piece_1, self.piece_2) for part in _pose_to_gripper(self.arm, d, body)])

    def extra_obs(self, d):
        """MimicGen's grip-site poses relative to the base and to each piece."""
        prefix, obs = self.arm.prefix, {}
        for name, body in (("base", self.base), ("piece_1", self.piece_1), ("piece_2", self.piece_2)):
            obs[f"{prefix}eef_pos_rel_{name}"], obs[f"{prefix}eef_quat_rel_{name}"] = _gripper_in(self.arm, d, body)
        return obs

    def _first_assembled(self, d):
        return (jnp.linalg.norm(d.xpos[self.piece_1, :2] - d.xpos[self.base, :2]) < self.xy_thresh) & ~self.grasp(d, self.only_1)

    def success(self, d):
        base, p1, p2 = d.xpos[self.base], d.xpos[self.piece_1], d.xpos[self.piece_2]
        stacked = (jnp.linalg.norm(p1[:2] - p2[:2]) < self.xy_thresh) & (jnp.abs(p2[2] - base[2] - self.piece_2_height) < self.z_thresh)
        return self._first_assembled(d) & stacked & ~self.grasp(d, self.only_2)

    def reward(self, d):
        return self.success(d).astype(jnp.float32)

    def _hole_margin(self, d):
        """How far the base's square hole clears piece 1's slab in the base's frame: positive once the slab fits through."""
        yaw = _yaw(d, self.piece_1) - _yaw(d, self.base)
        extent = self.slab_half * (jnp.abs(jnp.cos(yaw)) + jnp.abs(jnp.sin(yaw)))
        off = (d.xmat[self.base].reshape(3, 3).T @ (d.xpos[self.piece_1] - d.xpos[self.base]))[:2]
        return self.hole_half - extent - jnp.max(jnp.abs(off))

    def _slot_margin(self, d):
        """How far piece 2's slot (along its y axis) clears piece 1's ridge across the slot: positive once it fits over."""
        yaw = _yaw(d, self.piece_1) - _yaw(d, self.piece_2)
        extent = self.ridge_half[0] * jnp.abs(jnp.cos(yaw)) + self.ridge_half[1] * jnp.abs(jnp.sin(yaw))
        off = (d.xmat[self.piece_2].reshape(3, 3).T @ (d.xpos[self.piece_1] - d.xpos[self.piece_2]))[0]
        return self.slot_half - extent - jnp.abs(off)

    def _released(self, d, piece, target, place, margin, held):
        """
        Whether `piece`, let go of, is centered over `target` within `release_radius` and rests at height `place` or drops
        toward it from within `release_height` above, fitting (positive `margin`) unless already at `place`.
        """
        pos = d.xpos[piece]
        height = pos[2] - place
        centered = jnp.linalg.norm(pos[:2] - target[:2]) < self.release_radius
        resting = jnp.abs(height) < self.seat_slack
        dropping = (height > 0.) & (height < self.release_height) & (margin > 0.)
        return ~held & centered & (resting | dropping)

    def _placing(self, d, piece, grip, target, place, clearance, margin, mask):
        """
        Progress of placing `piece`, and whether it rests at its goal: reaching its grip geom from above, holding it,
        lifting it to `clearance`, carrying it over `target` and fitting it (level, centered and turned so it fits), and
        lowering it to `place`, with horizontal distance to `target` counted beyond `goal_tolerance`. Released at its goal
        counts as held.
        """
        pos = d.xpos[piece]
        held = self.grasp.held(d, mask)
        released = self._released(d, piece, target, place, margin, held)
        offset = jnp.maximum(jnp.linalg.norm(pos[:2] - target[:2]) - self.goal_tolerance, 0.)
        up, over, down = _lift_and_carry(pos[2], offset, place, clearance, self.carry_scale)
        level = jnp.clip(d.xmat[piece].reshape(3, 3)[2, 2], 0., 1.)
        fits = _fine(1. - level, _LEVEL_SCALE, _LEVEL_FINE) * _fine(jnp.maximum(-margin, 0.), _FINE_ALIGN, _FIT_FINE)
        reach = jnp.where(released, 1., _reach(self.arm, d, d.geom_xpos[grip]))
        grip_ = _grip(self.arm, self.grasp, d, held | released, d.geom_xpos[grip], mask)
        progress = _progress(reach, grip_, held | released, up, 0.5 * (over + up * fits), down)
        return progress, released & (jnp.abs(pos[2] - place) < self.seat_slack)

    def progress(self, d):
        """Half for seating piece 1 in the base; half, once piece 1 rests in the base, for setting piece 2 on it."""
        base, p1 = d.xpos[self.base], d.xpos[self.piece_1]
        first, seated = self._placing(
            d, self.piece_1, self.grip_1, base, base[2] + self.rest_1, base[2] + self.clearance_1, self._hole_margin(d), self.only_1,
        )
        second, _ = self._placing(
            d, self.piece_2, self.grip_2, p1, base[2] + self.rest_2, base[2] + self.clearance_2, self._slot_margin(d), self.only_2,
        )
        return 0.5 * first + 0.5 * jnp.where(seated, second, 0.)


class CoffeePreparation(Coffee):
    """
    MimicGen's CoffeePreparation: the mug set upright on the coffee machine's base plate, the pod taken from the drawer
    into the machine's holder, and the lid closed over it.
    """
    def __init__(self, spec, m, arms):
        super().__init__(spec, m, arms)
        self.drawer, self.mug = m.body(spec["drawer"]).id, m.body(spec["mug"]).id
        self.slide = int(m.jnt_qposadr[m.joint(spec["drawer_slide"]).id])
        self.handle = m.geom(spec["drawer_handle"]).id
        self.drawer_open = spec["drawer_open"]
        self.lid_ready = spec["lid_ready"]
        self.mug_upright_tol = spec["mug_upright_tol"]
        self.mug_grip_offset = jnp.asarray(spec["mug_grip_offset"])
        self.mug_place_offset = jnp.asarray(spec["mug_place_offset"])
        self.mug_clearance = spec["mug_clearance"]
        self.mug_carry_scale = spec["mug_carry_scale"]
        self.mug_grasp = _Grasp(m, spec["mug_pad_geoms"], [spec["mug_geoms"]])
        self.mug_on_base = ContactTest(m, _ids(m.geom, [spec["base_geom"]]), _ids(m.geom, spec["mug_geoms"]))

    def object_obs(self, d, cache):
        bodies = (self.pod, self.machine, self.holder, self.lid, self.drawer, self.mug)
        poses = [part for body in bodies for part in _pose_to_gripper(self.arm, d, body)]
        return jnp.concatenate([*poses, d.qpos[self.hinge][None], d.qpos[self.slide][None]])

    def _mug_placed(self, d):
        upright = 1. - d.xmat[self.mug].reshape(3, 3)[2, 2] < self.mug_upright_tol
        return upright & self.mug_on_base(d)

    def success(self, d):
        return super().success(d) & self._mug_placed(d)

    def _rotated(self, d, body, offset):
        return d.xpos[body] + quat2mat(wxyz_to_xyzw(d.xquat[body])) @ offset

    def _mug_progress(self, d):
        """Reaching the mug's rim, holding it, carrying it onto the base plate under the holder; 1 once released there."""
        mug = d.xpos[self.mug]
        held = self.mug_grasp.held(d)
        done = ~held & self._mug_placed(d)
        grip_point = self._rotated(d, self.mug, self.mug_grip_offset)
        target = self._rotated(d, self.holder, self.mug_place_offset)
        offset = jnp.linalg.norm(mug[:2] - target[:2])
        reach = _reach(self.arm, d, grip_point)
        grip = _grip(self.arm, self.mug_grasp, d, held, grip_point)
        carry = _lift_and_carry(mug[2], offset, target[2], target[2] + self.mug_clearance, self.mug_carry_scale)
        return jnp.where(done, 1., _progress(reach, grip, held, *carry)), done

    def _pod_progress(self, d):
        """`Coffee.progress`'s pod half: reaching, holding, carrying the pod into the holder; held once released in it."""
        pod, holder = d.xpos[self.pod], d.xpos[self.holder]
        held = self.grasp.held(d)
        released = ~held & self._pod_in_holder(d)
        place = holder[2] + self.pod_rest_height
        offset = jnp.linalg.norm(pod[:2] - holder[:2])
        reach = jnp.where(released, 1., _reach(self.arm, d, pod))
        grip = _grip(self.arm, self.grasp, d, held | released, pod)
        carry = _lift_and_carry(pod[2], offset, place, holder[2] + self.holder_half[2] + 2. * self.pod_half[2], self.carry_scale)
        return _progress(reach, grip, held | released, *carry), held, released

    def progress(self, d):
        """
        In the demonstrations' order, each stage's reaching counting once the stages before it are done: placing the
        mug (0.25), opening the lid (0.1), pulling the drawer open (0.1), putting the pod in the holder (0.35), and
        closing the lid over it (0.2). Opening the lid and the drawer count as done once the pod is held or in place.
        """
        mug, mug_done = self._mug_progress(d)
        pod, pod_held, pod_in = self._pod_progress(d)
        pod_taken = pod_held | pod_in

        hinge = d.qpos[self.hinge]
        opened = jnp.clip(hinge / self.lid_ready, 0., 1.)
        lid_ready = opened >= 1.
        reach_lid = jnp.where(lid_ready, 1., jnp.where(mug_done, _reach(self.arm, d, d.xpos[self.lid]), 0.))
        lid = jnp.where(pod_taken, 1., 0.25 * reach_lid + 0.75 * opened)

        pulled = jnp.clip(-d.qpos[self.slide] / self.drawer_open, 0., 1.)
        drawer_ready = pulled >= 1.
        reach_handle = jnp.where(drawer_ready, 1., jnp.where(lid_ready, _reach(self.arm, d, d.geom_xpos[self.handle]), 0.))
        drawer = jnp.where(pod_taken, 1., 0.25 * reach_handle + 0.75 * pulled)

        pod = jnp.where(pod_taken | (lid_ready & drawer_ready), pod, 0.)
        closed = _ramp(jnp.maximum(hinge - self.lid_closed, 0.), self.hinge_open - self.lid_closed)
        close = jnp.where(pod_in, closed, 0.)
        return 0.25 * mug + 0.1 * lid + 0.1 * drawer + 0.35 * pod + 0.2 * close


TASKS = {
    "lift": Lift,
    "can": PickPlace,
    "square": NutAssembly,
    "tool_hang": ToolHang,
    "transport": TwoArmTransport,
    "stack_d0": Stack,
    "stack_d1": Stack,
    "stack_three_d0": StackThree,
    "stack_three_d1": StackThree,
    "coffee_d0": Coffee,
    "coffee_d1": Coffee,
    "coffee_d2": Coffee,
    "coffee_preparation_d0": CoffeePreparation,
    "coffee_preparation_d1": CoffeePreparation,
    "threading_d0": Threading,
    "mug_cleanup_d0": MugCleanup,
    "mug_cleanup_d1": MugCleanup,
    "square_d0": MimicGenNutAssembly,
    "square_d1": MimicGenNutAssembly,
    "square_d2": MimicGenNutAssembly,
    "nut_assembly_d0": MimicGenNutAssembly,
    "pick_place_d0": MimicGenPickPlace,
    "hammer_cleanup_d0": HammerCleanup,
    "hammer_cleanup_d1": HammerCleanup,
    "threading_d1": Threading,
    "threading_d2": Threading,
    "three_piece_assembly_d0": ThreePieceAssembly,
    "three_piece_assembly_d1": ThreePieceAssembly,
    "three_piece_assembly_d2": ThreePieceAssembly,
    "kitchen_d0": Kitchen,
    "kitchen_d1": Kitchen,
}
