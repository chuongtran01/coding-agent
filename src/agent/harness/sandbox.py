"""Run a task in a fresh, isolated Docker container per attempt."""

from __future__ import annotations

import io
import posixpath
import shlex
import tarfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import docker  # type: ignore[import-untyped]
from docker.errors import ImageNotFound, NotFound  # type: ignore[import-untyped]
from pydantic import BaseModel

from agent.harness.dataset import Task

# SWE-bench publishes x86_64 images only; on Apple Silicon they run emulated.
PLATFORM = "linux/amd64"
WORKDIR = "/testbed"
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate testbed"
CONTAINER_LABEL = "agent.harness.sandbox"

DEFAULT_MEM_LIMIT = "4g"
DEFAULT_CPUS = 2.0
DEFAULT_PIDS_LIMIT = 1024
DEFAULT_TIMEOUT_S = 300
# The container's main process; an orphaned container (host process killed) stops by
# itself after this long instead of running forever.
DEFAULT_LIFETIME_S = 4 * 60 * 60

# Exit codes from coreutils `timeout`: 124 after SIGTERM, 137 after the SIGKILL follow-up.
_TIMEOUT_EXIT_CODES = (124, 137)
# Published images add one "SWE-bench" commit on top of base_commit that only changes file modes.
_SWEBENCH_SETUP_SUBJECT = "SWE-bench"


class SandboxError(RuntimeError):
    """The sandbox could not be started or is not in the expected state."""


class ExecResult(BaseModel):
    """Outcome of one command run inside the sandbox."""

    exit_code: int
    output: str  # stdout and stderr, interleaved
    runtime_s: float
    timed_out: bool


class Sandbox:
    """A running container for one attempt at one task.

    Create it with `Sandbox.start(task)`; the container is removed when the block exits.
    Nothing from the host is mounted, so the container's filesystem is all the agent sees.
    Networking is off unless asked for, so the agent cannot download the upstream fix;
    evaluation turns it on because the eval scripts reinstall packages from PyPI.
    """

    def __init__(self, task: Task, container: Any) -> None:
        """Wrap an already-running container; use `Sandbox.start` instead."""
        self.task = task
        self._container = container

    @classmethod
    @contextmanager
    def start(
        cls,
        task: Task,
        *,
        mem_limit: str = DEFAULT_MEM_LIMIT,
        cpus: float = DEFAULT_CPUS,
        pids_limit: int = DEFAULT_PIDS_LIMIT,
        network: bool = False,
        lifetime_s: int = DEFAULT_LIFETIME_S,
    ) -> Generator[Sandbox, None, None]:
        """Start a container for `task` at its base commit, and remove it afterwards."""
        client = docker.from_env()
        _ensure_image(client, task.image)
        container = client.containers.run(
            task.image,
            command=f"sleep {lifetime_s}",
            detach=True,
            platform=PLATFORM,
            working_dir=WORKDIR,
            mem_limit=mem_limit,
            memswap_limit=mem_limit,  # no swap on top of the memory limit
            nano_cpus=int(cpus * 1e9),
            pids_limit=pids_limit,
            network_mode=None if network else "none",
            labels={CONTAINER_LABEL: task.instance_id},
        )
        try:
            sandbox = cls(task, container)
            sandbox._check_base_commit()
            yield sandbox
        finally:
            _remove(container)

    def exec(self, cmd: str, timeout: float = DEFAULT_TIMEOUT_S) -> ExecResult:
        """Run a shell command in the repo with the task's Python environment active."""
        wrapped = (
            f"{CONDA_ACTIVATE} && "
            f"timeout --kill-after=5 {timeout} bash -c {shlex.quote(cmd)}"
        )
        start = time.monotonic()
        exit_code, output = self._container.exec_run(
            ["bash", "-c", wrapped], workdir=WORKDIR
        )
        runtime_s = time.monotonic() - start
        return ExecResult(
            exit_code=exit_code,
            output=output.decode("utf-8", errors="replace"),
            runtime_s=runtime_s,
            timed_out=exit_code in _TIMEOUT_EXIT_CODES and runtime_s >= timeout,
        )

    def write_file(self, path: str, content: str) -> None:
        """Write a text file into the container, creating parent directories."""
        if not posixpath.isabs(path):
            path = posixpath.join(WORKDIR, path)
        directory, name = posixpath.split(path)
        mkdir = self.exec(f"mkdir -p {shlex.quote(directory)}", timeout=60)
        if mkdir.exit_code != 0:
            raise SandboxError(f"failed to create {directory}: {mkdir.output}")

        data = content.encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
        if not self._container.put_archive(directory, buf.getvalue()):
            raise SandboxError(f"failed to write {path}")

    def _check_base_commit(self) -> None:
        result = self.exec("git rev-parse HEAD HEAD~1 && git log -1 --format=%s", 60)
        lines = result.output.strip().splitlines()
        if result.exit_code != 0 or len(lines) < 3:
            raise SandboxError(
                f"{self.task.instance_id}: git check failed:\n{result.output}"
            )
        head, parent, subject = (line.strip() for line in lines[-3:])
        base = self.task.base_commit
        if head == base or (parent == base and subject == _SWEBENCH_SETUP_SUBJECT):
            return
        raise SandboxError(
            f"{self.task.instance_id}: expected base_commit {base}, got HEAD {head}"
        )


def remove_leftover_containers() -> int:
    """Remove sandbox containers left behind by a crashed run; return how many."""
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"label": CONTAINER_LABEL})
    for container in containers:
        _remove(container)
    return len(containers)


def _ensure_image(client: Any, image: str) -> None:
    try:
        client.images.get(image)
    except ImageNotFound:
        repository, tag = image.rsplit(":", 1)
        client.images.pull(repository, tag=tag, platform=PLATFORM)


def _remove(container: Any) -> None:
    try:
        container.remove(force=True)
    except NotFound:
        pass
