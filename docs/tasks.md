# Tasks

The five robomimic simulation tasks as POMDPs, made with `jax_pomdps.make("robomimic.env:<name>")`. Every task
is robosuite's: its scene, robots, controller, observations, reward, and success check, re-expressed in JAX on
MuJoCo Warp. The quotes are robosuite v1.5's docstrings; the images show the first and last state of each
task's first released demonstration.

Names follow [jax_gym](https://github.com/mishmish66/jax-gym)'s scheme: dashes join words, and slashes add
variants. Each task `name` observes the low-dim observations of the robomimic datasets, and also exists as

- `name/prp`, observing each robot's proprioception alone (the `robot<i>_` keys);
- `name/pix`, observing the cameras of robomimic's image datasets alone, as `<camera>_image` uint8 images:
  `agentview` and `robot0_eye_in_hand` at 84×84 for Lift, Can, and Square, `shouldercamera0`,
  `shouldercamera1`, `robot0_eye_in_hand`, and `robot1_eye_in_hand` at 84×84 for Transport, and `sideview` and
  `robot0_eye_in_hand` at 240×240 for Tool Hang;
- `name/pix-prp`, observing those images and proprioception, as robomimic's image policies do.

`camera_names`, `camera_height`, and `camera_width` change the cameras, and every other argument of
`RobomimicPOMDP` (such as `max_worlds` or `reward_shaping`) passes through.

All tasks share these conventions:

- **Actions** are robosuite's OSC_POSE deltas for each arm, in the world frame, followed by its gripper
  command: 6 values in [-1, 1] scaled to ±5 cm and ±0.5 rad, and one open/close value; Transport has two arms,
  so 14 values.
- **Control** runs at 20 Hz, robosuite's rate for the robomimic datasets: each step is 10 substeps of 5 ms,
  and the OSC controller recomputes the arm torques at every substep.
- **Observations** are a dictionary keyed like robosuite's: per robot, `robot<i>_joint_pos`, `_joint_pos_cos`,
  `_joint_pos_sin`, `_joint_vel`, `_eef_pos`, `_eef_quat`, `_eef_quat_site`, `_gripper_qpos`, and
  `_gripper_qvel`, and `object`, the task's object observations, in robosuite's order. With `camera_names`,
  `<camera>_image` (and `<camera>_depth`) are added. Quaternions are (x, y, z, w).
- **Reward** is the sparse reward of the datasets: 1 when the task succeeds, 0 otherwise (robosuite's
  normalized sparse reward with `reward_scale` 1). With `reward_shaping=True` it is robosuite's shaped reward,
  which robosuite has for Lift, Can, and Square; Tool Hang and Transport keep the sparse reward.
- **Episodes** end on success (`done`); there is no time limit.
- **Initial states** are drawn from 1024 states sampled from robosuite's reset distribution.

## `lift`

<img src="images/lift.png" alt="Lift: the first and last state of demo 0" loading="lazy">

> This class corresponds to the lifting task for a single robot arm.
>
> Sparse un-normalized reward: a discrete reward of 2.25 is provided if the cube is lifted
>
> — robosuite v1.5, `robosuite/environments/manipulation/lift.py`

A Panda arm lifts a cube from a table. **Success**: the cube's center is more than 4 cm above the table top.
**Object observation** (10): the cube's position and quaternion, and its position relative to the gripper.
**Shaped reward**: 1 on success, else (`1 - tanh(10 d)` + 0.25 while both finger pads touch the cube) / 2.25,
with `d` the distance from the gripper to the cube.

## `can`

<img src="images/can.png" alt="Can: the first and last state of demo 0" loading="lazy">

> Easier version of task - place one can into its bin.
>
> Sparse un-normalized reward: a discrete reward of 1.0 per object if it is placed in its correct bin
>
> — robosuite v1.5, `robosuite/environments/manipulation/pick_place.py` (`PickPlaceCan`)

A Panda arm moves a can from one bin into its compartment of the other. **Success**: the can is inside its
compartment, below the bin's rim, and the gripper has let go of it. **Object observation** (14): the can's
position and quaternion relative to the gripper, then in the world.
**Shaped reward**: 1 once the can is placed, else the largest of robosuite's staged rewards: reaching
(`0.1 (1 - tanh(10 d))`, `d` the distance from the gripper to the can), grasping (0.35 while both finger pads
touch it), lifting (0.35 to 0.5 while grasped, rising to 25 cm above the bin), and hovering (up to 0.7, nearing
the center of its compartment). As in robosuite, the parked milk, bread, and cereal boxes take part, so any
grasp earns the full lifting reward.

## `square`

<img src="images/square.png" alt="Square: the first and last state of demo 0" loading="lazy">

> Easier version of task - place one square nut into its peg.
>
> Sparse un-normalized reward: a discrete reward of 1.0 per nut if it is placed around its correct peg
>
> — robosuite v1.5, `robosuite/environments/manipulation/nut_assembly.py` (`NutAssemblySquare`)

A Panda arm picks up a square nut by its handle and slides it down over a square peg. **Success**: the nut is
within 3 cm of the peg horizontally, low on the peg, and the gripper has let go of it. **Object observation**
(14): the nut's position and quaternion relative to the gripper, then in the world.
**Shaped reward**: 1 once the nut is placed, else the largest of robosuite's staged rewards: reaching
(`0.1 (1 - tanh(10 d))`, `d` the distance from the gripper to the nut's handle), grasping (0.35 while both
finger pads touch it), lifting (0.35 to 0.5 while grasped, rising to 20 cm above the table), and hovering (up to
0.7, nearing the peg). As in robosuite, the parked round nut takes part, so any grasp earns the full lifting
reward.

## `transport`

<img src="images/transport.png" alt="Transport: the first and last state of demo 0" loading="lazy">

> This class corresponds to the transport task for two robot arms, requiring a payload to be transported from
> an initial bin into a target bin, while removing trash from the target bin to a trash bin.
>
> Sparse un-normalized reward: a discrete reward of 1.0 is provided when the payload is in the target bin and
> the trash is in the trash bin
>
> — robosuite v1.5, `robosuite/environments/manipulation/two_arm_transport.py`

Two Panda arms facing each other: one lifts a lid and takes a hammer out of its bin and hands it over; the
other clears a piece of trash out of the target bin and places the hammer in it. **Success**: the hammer
touches the target bin's base and the trash touches the trash bin's base. **Object observation** (41): the
poses of the hammer, the trash, and the lid handle, the bin positions, both contact flags, and the objects'
positions relative to the grippers.

## `tool-hang`

<img src="images/tool_hang.png" alt="Tool Hang: the first and last state of demo 0" loading="lazy">

> This class corresponds to the tool hang task for a single robot arm.
>
> Check if tool is hung on frame correctly and frame is assembled coorectly as well.
>
> — robosuite v1.5, `robosuite/environments/manipulation/tool_hang.py`

A Panda arm inserts an L-shaped frame into a stand, then hangs a wrench-like tool on the frame's hook by its
ring. The tightest task: the frame and the ring leave millimeters of clearance. **Success**: the frame stands
upright in the stand's hole, and the tool hangs from the hook without touching the gripper. **Object
observation** (44): the poses of the stand, the frame's hook, and the tool, each relative to the gripper and in
the world, and whether the frame is assembled and the tool is on it.
