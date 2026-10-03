"""
Robomimic's manipulation tasks as jax_pomdps POMDPs simulated by MuJoCo Warp, registered with dense rewards as
"lift", "can", "square", "transport", and "tool-hang" and with the datasets' sparse rewards as "<name>/sparse", each
also observing proprioception ("/prp"), a Markov state ("/mkv"), images ("/pix"), or images and proprioception
("/pix-prp"):

    import jax_pomdps as pomdps
    env = pomdps.make("robomimic.env:tool-hang/sparse/pix-prp")

.. include:: ../../../docs/tasks.md
"""
from robomimic.env.pomdp import (
    PIXELS, TASK_OF_ENV_NAME, Pixels, RobomimicPOMDP, State, load_model, parked_joints, registry_name, task_spec,
)
from robomimic.env.tasks import TASKS

__all__ = [
    "PIXELS", "TASKS", "TASK_OF_ENV_NAME", "Pixels", "RobomimicPOMDP", "State", "load_model", "parked_joints",
    "registry_name", "task_spec",
]
