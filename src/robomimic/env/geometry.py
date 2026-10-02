"""
Rigid-body math in robosuite's conventions (quaternions are (x, y, z, w) with w >= 0) and exact
contact tests between named geoms of a simulated world.
"""
import itertools
from collections.abc import Sequence

import numpy as np

import jax
import jax.numpy as jnp
import mujoco
from mujoco.mjx._src import collision_convex
from mujoco.mjx._src import mesh as mjx_mesh
from mujoco.mjx._src.collision_types import ConvexInfo


def wxyz_to_xyzw(q):
    return jnp.concatenate([q[1:], q[:1]])


def mat2quat(R):
    """Unit quaternion (x, y, z, w) with w >= 0 of rotation matrix `R`."""
    tr = jnp.trace(R)
    candidates = jnp.stack([
        jnp.array([1. + tr, R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]),
        jnp.array([R[2, 1] - R[1, 2], 1. + R[0, 0] - R[1, 1] - R[2, 2], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]]),
        jnp.array([R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], 1. - R[0, 0] + R[1, 1] - R[2, 2], R[1, 2] + R[2, 1]]),
        jnp.array([R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1. - R[0, 0] - R[1, 1] + R[2, 2]]),
    ])
    q = candidates[jnp.argmax(jnp.array([tr, R[0, 0], R[1, 1], R[2, 2]]))]
    q = q / jnp.linalg.norm(q)
    q = jnp.where(q[0] < 0., -q, q)
    return wxyz_to_xyzw(q)


def quat2mat(q):
    """Rotation matrix of quaternion `q` given as (x, y, z, w)."""
    x, y, z, w = q / jnp.linalg.norm(q)
    return jnp.array([
        [1. - 2. * (y * y + z * z), 2. * (x * y - z * w), 2. * (x * z + y * w)],
        [2. * (x * y + z * w), 1. - 2. * (x * x + z * z), 2. * (y * z - x * w)],
        [2. * (x * z - y * w), 2. * (y * z + x * w), 1. - 2. * (x * x + y * y)],
    ])


def relative_pose(pos, rot, frame_pos, frame_rot):
    """Position and quaternion (x, y, z, w) of the pose (`pos`, `rot`) expressed in the frame (`frame_pos`, `frame_rot`)."""
    return frame_rot.T @ (pos - frame_pos), mat2quat(frame_rot.T @ rot)


def _skew(v):
    return jnp.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def axisangle2mat(v):
    angle = jnp.linalg.norm(v)
    nonzero = angle > 0.
    K = _skew(v / jnp.where(nonzero, angle, 1.))
    R = jnp.eye(3) + jnp.sin(angle) * K + (1. - jnp.cos(angle)) * (K @ K)
    return jnp.where(nonzero, R, jnp.eye(3))


# number of sides of the convex prism that stands in for a cylinder
_CYLINDER_PRISM_SIDES = 16


def _convex_arrays(vert, face):
    """MJX convex arrays (vertices, face vertices, face normals, edges, edge face normals) of a convex polytope."""
    face_normal = mjx_mesh._get_face_norm(vert, face)
    edge, edge_face_normal = mjx_mesh._get_edge_normals(face, face_normal)
    return vert, vert[face], face_normal, edge, edge_face_normal


def _unit_prism(sides):
    """Convex prism approximating the cylinder of radius 1 and half-height 1 along z, as MJX convex arrays."""
    angles = np.arange(sides) * 2. * np.pi / sides
    radius = 2. / (1. + np.cos(np.pi / sides))
    ring = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vert = np.concatenate([np.concatenate([ring, np.full((sides, 1), z)], axis=1) for z in (-1., 1.)])
    k = np.arange(sides)
    walls = np.stack([k, (k + 1) % sides, sides + (k + 1) % sides, sides + k], axis=1)
    walls = np.pad(walls, ((0, 0), (0, sides - 4)), mode="edge")
    face = np.concatenate([k[::-1][None], (sides + k)[None], walls])
    return _convex_arrays(vert, face)


_UNIT_PRISM = _unit_prism(_CYLINDER_PRISM_SIDES)


def _unit_box():
    """The box with half-extents 1, as MJX convex arrays."""
    vert = np.array(list(itertools.product((-1., 1.), (-1., 1.), (-1., 1.))))
    face = np.array([0, 4, 5, 1, 0, 2, 6, 4, 6, 7, 5, 4, 2, 3, 7, 6, 1, 5, 7, 3, 0, 1, 3, 2]).reshape(-1, 4)
    return _convex_arrays(vert, face)


_UNIT_BOX = _unit_box()


def _scaled(unit, scale):
    """MJX convex arrays of the unit polytope `unit` scaled by `scale`."""
    vert, face, face_normal, edge, edge_face_normal = unit
    return vert.astype(np.float32) * scale, face.astype(np.float32) * scale, face_normal, edge, edge_face_normal


def _convex_shape(m, g):
    """`ConvexInfo` of geom `g`, without a pose."""
    size = m.geom_size[g].astype(np.float32)
    match mujoco.mjtGeom(m.geom_type[g]):
        case mujoco.mjtGeom.mjGEOM_MESH:
            mesh = mjx_mesh.convex(m, m.geom_dataid[g])
            arrays = mesh.vert, mesh.face, mesh.face_normal, mesh.edge, mesh.edge_face_normal
        case mujoco.mjtGeom.mjGEOM_BOX:
            arrays = _scaled(_UNIT_BOX, size)
        case mujoco.mjtGeom.mjGEOM_CYLINDER:
            arrays = _scaled(_UNIT_PRISM, size[[0, 0, 1]])
        case t:
            raise NotImplementedError(f"contact tests for geom type {t.name}")
    return ConvexInfo(None, None, jnp.asarray(size), *(jnp.asarray(x) for x in arrays))


class ContactTest:
    """
    Whether any geom of one group touches any geom of another in a simulated world. Pairs that MuJoCo
    never collides (disjoint contype / conaffinity bits, or geoms of the same body) are skipped.
    """
    def __init__(self, m, geoms_a: Sequence[int], geoms_b: Sequence[int]):
        self._pairs = [
            (a, b) for a, b in itertools.product(geoms_a, geoms_b)
            if m.geom_bodyid[a] != m.geom_bodyid[b]
            and (m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a])
        ]
        if not self._pairs:
            raise ValueError("no collidable geom pairs")
        self._shapes = {g: _convex_shape(m, g) for pair in self._pairs for g in pair}
        self._boxes = {g for g in self._shapes if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX}
        self._margin = np.asarray(m.geom_margin)

    def __call__(self, d):
        """Whether the two geom groups are in contact in world data `d`."""
        touching = []
        for a, b in self._pairs:
            ia, ib = (self._shapes[g].replace(pos=d.geom_xpos[g], mat=d.geom_xmat[g].reshape(3, 3)) for g in (a, b))
            collide = collision_convex._box_box if {a, b} <= self._boxes else collision_convex._convex_convex
            dist, _, _ = collide(ia, ib)
            touching.append(jnp.min(dist) < max(self._margin[a], self._margin[b]))
        return jnp.any(jnp.stack(touching))
