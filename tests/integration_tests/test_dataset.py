import re

from agent.harness.dataset import DEV_TASK_IDS, load_dev_tasks, load_tasks


def test_load_full_split() -> None:
    tasks = load_tasks()
    assert len(tasks) == 300
    for task in tasks.values():
        assert re.fullmatch(r"[0-9a-f]{40}", task.base_commit)
        assert task.FAIL_TO_PASS
        assert task.gold_patch.startswith("diff --git")
        assert task.test_patch.startswith("diff --git")


def test_dev_tasks_exist() -> None:
    tasks = load_dev_tasks()
    assert [t.instance_id for t in tasks] == list(DEV_TASK_IDS)
