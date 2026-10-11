"""A stand-in for the Docker sandbox that runs commands with the host's bash."""

import subprocess
import time
from pathlib import Path

from agent.harness.sandbox import ExecResult


class LocalShell:
    """Runs commands with the host's bash in a temp dir, standing in for a Sandbox."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def exec(self, cmd: str, timeout: float = 60) -> ExecResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                exit_code=124, output="", runtime_s=timeout, timed_out=True
            )
        return ExecResult(
            exit_code=proc.returncode,
            output=proc.stdout.decode("utf-8", errors="replace"),
            runtime_s=time.monotonic() - start,
            timed_out=False,
        )

    def write_file(self, path: str, content: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)
