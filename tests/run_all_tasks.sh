#!/bin/bash
# Runs the test suite over every task, each task's tests in a pytest process of their own, so that GPU memory
# held by one task's simulations is released before the next; extra arguments go to pytest.
cd "$(dirname "$0")/.."
tasks=$(uv run python -c "from robomimic.env import TASKS; print(' '.join(TASKS))")
status=0
uv run pytest tests -q -p no:cacheprovider --task none "$@" || status=1
for task in $tasks; do
  echo "== $task"
  uv run pytest tests -q -p no:cacheprovider --task "$task" "$@" || status=1
done
exit $status
