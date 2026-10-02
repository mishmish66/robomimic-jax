"""
Robomimic's manipulation tasks as jax_pomdps POMDPs simulated by MuJoCo Warp, registered as "robomimic/lift",
"robomimic/can", "robomimic/square", "robomimic/transport", and "robomimic/tool_hang":

    import jax_pomdps as pomdps
    env = pomdps.make("robomimic.env:robomimic/lift")

.. include:: ../../../docs/tasks.md
"""
from robomimic.env.pomdp import TASK_OF_ENV_NAME, RobomimicPOMDP, State, load_model, parked_joints, task_spec
from robomimic.env.tasks import TASKS

__all__ = ["TASKS", "TASK_OF_ENV_NAME", "RobomimicPOMDP", "State", "load_model", "parked_joints", "task_spec"]
