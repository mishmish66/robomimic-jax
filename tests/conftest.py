from pathlib import Path

import jax

# compiled programs are kept across tests and runs
jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/robomimic-jax/xla").expanduser()))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# tasks of the default run: two-arm, MimicGen with per-reset fixtures and task memory, and the Sawyer arm
FAST_TASKS = ("lift", "transport", "square_d1", "kitchen_d1", "nut_assembly_d0")


def pytest_addoption(parser):
    parser.addoption("--all-tasks", action="store_true", help="run the per-task tests for every task, not FAST_TASKS")
    parser.addoption("--task", help="run only the tests of this task, or with none, only the tests of no task")


def _task(item):
    params = getattr(item, "callspec", None)
    params = params.params if params else {}
    return params.get("task", params.get("env"))


def pytest_collection_modifyitems(config, items):
    """Per-task tests of FAST_TASKS only, without --all-tasks, or of the one task of --task; each task's tests run together."""
    only = config.getoption("--task")
    if only is not None:
        skipped = [item for item in items if (_task(item) or "none") != only]
    elif not config.getoption("--all-tasks"):
        skipped = [item for item in items if _task(item) is not None and _task(item) not in FAST_TASKS]
    else:
        skipped = []
    if skipped:
        config.hook.pytest_deselected(items=skipped)
        items[:] = [item for item in items if item not in skipped]
    order = {item: i for i, item in enumerate(items)}
    items.sort(key=lambda item: (_task(item) or "", order[item]))
