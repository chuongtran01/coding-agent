"""Needs Docker and pulls each dev task's image (about 3 GB each) on first run."""

from typing import Any

import docker  # type: ignore[import-untyped]
import pytest

from agent.harness.dataset import Task, load_dev_tasks
from agent.harness.sandbox import CONTAINER_LABEL, Sandbox

DEV_TASKS = load_dev_tasks()


def _sandbox_containers() -> list[Any]:
    client = docker.from_env()
    containers: list[Any] = client.containers.list(
        all=True, filters={"label": CONTAINER_LABEL}
    )
    return containers


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_start_at_base_commit_and_clean_up(task: Task) -> None:
    with Sandbox.start(task) as sb:
        result = sb.exec("git rev-parse HEAD~1 HEAD")
        assert result.exit_code == 0
        assert task.base_commit in result.output.split()
        assert (
            sb.exec("python -c 'import sys; print(sys.prefix)'")
            .output.strip()
            .endswith("envs/testbed")
        )
    assert _sandbox_containers() == []


def test_timeout_kills_long_command() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        result = sb.exec("sleep 999", timeout=3)
    assert result.timed_out
    assert result.runtime_s < 15


def test_write_file_round_trip() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        sb.write_file("/tmp/agent/hello.txt", "héllo\nworld\n")
        result = sb.exec("cat /tmp/agent/hello.txt")
    assert result.output == "héllo\nworld\n"


def test_container_removed_after_exception() -> None:
    with pytest.raises(RuntimeError):
        with Sandbox.start(DEV_TASKS[0]):
            raise RuntimeError("boom")
    assert _sandbox_containers() == []


def test_limits_applied() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        sb._container.reload()
        host_config = sb._container.attrs["HostConfig"]
    assert host_config["Memory"] == 4 * 1024**3
    assert host_config["NanoCpus"] == 2 * 10**9
    assert host_config["PidsLimit"] == 1024
    assert host_config["Binds"] in (None, [])
