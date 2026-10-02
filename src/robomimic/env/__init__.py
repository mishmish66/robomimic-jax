"""
Robomimic's manipulation tasks as jax_pomdps POMDPs simulated by MuJoCo Warp, registered as "lift", "can",
"square", "transport", and "tool-hang", each also as "<name>/prp", "<name>/pix", and "<name>/pix-prp":

    import jax_pomdps as pomdps
    env = pomdps.make("robomimic.env:tool-hang/pix-prp")

.. include:: ../../../docs/tasks.md
"""
from robomimic.env.pomdp import (
    PIXELS, TASK_OF_ENV_NAME, RobomimicPOMDP, State, load_model, parked_joints, registry_name, task_spec,
)
from robomimic.env.tasks import TASKS

__all__ = [
    "PIXELS", "TASKS", "TASK_OF_ENV_NAME", "RobomimicPOMDP", "State", "load_model", "parked_joints", "registry_name",
    "task_spec",
]
