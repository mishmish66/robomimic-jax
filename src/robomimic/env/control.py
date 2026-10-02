"""
Robosuite's OSC_POSE arm controller and Panda gripper controller as JAX functions of simulated world data.
"""
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from mujoco import mjx

from robomimic.env.geometry import axisangle2mat, mat2quat, quat2mat, wxyz_to_xyzw

# stiffness of the nullspace pull toward the initial joint positions
_NULLSPACE_KP = 10.
# sign of each finger's control change as the gripper closes
_GRIP_SIGN = np.array([-1., 1.], np.float32)


def _orientation_error(desired, current):
    return 0.5 * (
        jnp.cross(current[:, 0], desired[:, 0])
        + jnp.cross(current[:, 1], desired[:, 1])
        + jnp.cross(current[:, 2], desired[:, 2])
    )


class _Scaling(NamedTuple):
    """Robosuite's affine map of controller inputs, clipped to [in_min, in_max], to outputs."""
    in_min: jax.Array
    in_max: jax.Array
    in_mid: jax.Array
    ratio: jax.Array
    out_mid: jax.Array

    @classmethod
    def from_spec(cls, spec):
        in_min, in_max = np.asarray(spec["input_min"], float), np.asarray(spec["input_max"], float)
        out_min, out_max = np.asarray(spec["output_min"], float), np.asarray(spec["output_max"], float)
        ratio = np.abs(out_max - out_min) / np.abs(in_max - in_min)
        return cls(*(
            jnp.asarray(x, jnp.float32) for x in (in_min, in_max, (in_max + in_min) / 2., ratio, (out_max + out_min) / 2.)
        ))

    def __call__(self, x):
        return (jnp.clip(x, self.in_min, self.in_max) - self.in_mid) * self.ratio + self.out_mid


def _operational_space_inverses(mass_factor, J):
    """
    Operational-space inertia pinv(J M^-1 J^T) and dynamically consistent inverse M^-1 J^T pinv(J M^-1 J^T) of
    Jacobian `J`, for the Cholesky factor L of M. Both come from the pseudo-inverse of A = J L^-T, since
    A A^T = J M^-1 J^T and A has the square root of its condition number.
    """
    A = jax.scipy.linalg.solve_triangular(mass_factor, J.T, lower=True).T
    A_pinv = jnp.linalg.pinv(A)
    return A_pinv.T @ A_pinv, jax.scipy.linalg.solve_triangular(mass_factor.T, A_pinv, lower=False)


def _joint_addresses(m, names):
    """qpos and dof addresses of the named joints."""
    joints = [m.joint(n) for n in names]
    return np.array([j.qposadr[0] for j in joints]), np.array([j.dofadr[0] for j in joints])


class Arm:
    """Controller parameters and model indices for one OSC_POSE arm and its Panda gripper."""
    def __init__(self, spec, m):
        """Arm of `spec`, a task's "arms" entry in tasks.json, in model `m`."""
        self.prefix = spec["prefix"]
        self.arm_action = slice(*spec["arm_action"])
        self.grip_action = slice(*spec["grip_action"])
        self.arm_scale = _Scaling.from_spec(spec["arm_scale"])
        self.grip_scale = _Scaling.from_spec(spec["grip_scale"])

        self.site = m.site(spec["site"]).id
        self.site_body = int(m.site_bodyid[self.site])
        self.body = m.body(spec["body"]).id
        self.qpos, self.qvel = _joint_addresses(m, spec["joints"])
        self.grip_qpos, self.grip_qvel = _joint_addresses(m, spec["grip_joints"])
        self.kp = jnp.asarray(spec["kp"], jnp.float32)
        self.kd = jnp.asarray(spec["kd"], jnp.float32)

        # address in the sparse mass matrix of each entry of the arm block
        assert np.array_equal(m.M_rowadr, np.cumsum(m.M_rownnz) - m.M_rownnz)
        rows = np.repeat(np.arange(m.nv), m.M_rownnz)
        adrs = np.arange(len(rows))
        n = len(self.qvel)
        position = np.full(m.nv, -1)
        position[self.qvel] = np.arange(n)
        r, c = position[rows], position[m.M_colind]
        keep = (r >= 0) & (c >= 0)
        adr = np.full((n, n), -1)
        adr[r[keep], c[keep]] = adr[c[keep], r[keep]] = adrs[keep]
        if np.any(adr < 0):
            raise ValueError("arm mass matrix block is not dense")
        self._mass_adr = adr

        self.actuators = np.array([m.actuator(n).id for n in spec["actuators"]])
        self.grip_actuators = np.array([m.actuator(n).id for n in spec["grip_actuators"]])
        self.arm_ctrl_lo, self.arm_ctrl_hi = jnp.asarray(m.actuator_ctrlrange[self.actuators].T, jnp.float32)
        self.grip_ctrl_lo, self.grip_ctrl_hi = jnp.asarray(m.actuator_ctrlrange[self.grip_actuators].T, jnp.float32)
        self.grip_speed = spec["grip_speed"]

    def _mass(self, d):
        """Joint-space mass matrix of the arm's dofs."""
        return d._impl.M[self._mass_adr]

    def eef_pos(self, d):
        """Position of the grip site."""
        return d.site_xpos[self.site]

    def eef_rot(self, d):
        """Orientation of the eef body."""
        return quat2mat(wxyz_to_xyzw(d.xquat[self.body]))

    def goal(self, d, action):
        """World-frame OSC goal pose for this arm's slice of `action`, given kinematics `d`."""
        delta = self.arm_scale(action[self.arm_action])
        ref_pos = d.site_xpos[self.site]
        ref_ori = d.site_xmat[self.site].reshape(3, 3)
        frame = jnp.eye(3)  # orientation of the frame the action deltas are expressed in
        return ref_pos + frame @ delta[:3], frame @ axisangle2mat(delta[3:6]) @ frame.T @ ref_ori

    def torques(self, m, d, goal_pos, goal_ori, q0):
        """OSC joint torques toward the goal pose, with nullspace pull toward joint positions `q0`."""
        dofs = self.qvel
        ref_pos = d.site_xpos[self.site]
        ref_ori = d.site_xmat[self.site].reshape(3, 3)
        jacp, jacr = mjx.jac(m, d, ref_pos, self.site_body)
        vel_pos = jacp.T @ d.qvel
        vel_ori = jacr.T @ d.qvel
        J_pos = jacp[dofs].T
        J_ori = jacr[dofs].T
        J_full = jnp.concatenate([J_pos, J_ori], axis=0)
        mass = self._mass(d)
        joint_pos = d.qpos[self.qpos]
        joint_vel = d.qvel[dofs]

        force = self.kp[:3] * (goal_pos - ref_pos) - self.kd[:3] * vel_pos
        torque = self.kp[3:] * _orientation_error(goal_ori, ref_ori) - self.kd[3:] * vel_ori

        mass_factor = jnp.linalg.cholesky(mass)
        _, jbar = _operational_space_inverses(mass_factor, J_full)
        lambda_pos, _ = _operational_space_inverses(mass_factor, J_pos)
        lambda_ori, _ = _operational_space_inverses(mass_factor, J_ori)
        wrench = jnp.concatenate([lambda_pos @ force, lambda_ori @ torque])
        nullspace = jnp.eye(J_full.shape[1]) - jbar @ J_full

        pose_torques = mass @ (_NULLSPACE_KP * (q0 - joint_pos) - 2. * jnp.sqrt(_NULLSPACE_KP) * joint_vel)
        torques = J_full.T @ wrench + d.qfrc_bias[dofs] + nullspace.T @ pose_torques
        return jnp.clip(torques, self.arm_ctrl_lo, self.arm_ctrl_hi)

    def step_grip(self, grip, action):
        """Advance the Panda gripper's open / close state by one policy step."""
        return jnp.clip(grip + _GRIP_SIGN * self.grip_speed * jnp.sign(action[self.grip_action]), -1., 1.)

    def grip_ctrl(self, grip):
        """Gripper actuator controls at open/close state `grip`."""
        goal = self.grip_scale(grip)
        bias = 0.5 * (self.grip_ctrl_hi + self.grip_ctrl_lo)
        weight = 0.5 * (self.grip_ctrl_hi - self.grip_ctrl_lo)
        return jnp.clip(bias + weight * goal, self.grip_ctrl_lo, self.grip_ctrl_hi)

    def observe(self, d):
        """Robosuite's proprioceptive observations of the arm and its gripper."""
        joint_pos = d.qpos[self.qpos]
        obs = {
            "joint_pos": joint_pos,
            "joint_pos_cos": jnp.cos(joint_pos),
            "joint_pos_sin": jnp.sin(joint_pos),
            "joint_vel": d.qvel[self.qvel],
            "eef_pos": self.eef_pos(d),
            "eef_quat": wxyz_to_xyzw(d.xquat[self.body]),
            "eef_quat_site": mat2quat(d.site_xmat[self.site].reshape(3, 3)),
            "gripper_qpos": d.qpos[self.grip_qpos],
            "gripper_qvel": d.qvel[self.grip_qvel],
        }
        return {f"{self.prefix}{k}": v for k, v in obs.items()}
