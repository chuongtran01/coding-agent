"""The agent's tools: read, write, edit, bash and submit, run inside a sandbox.

Every tool returns a string for the model. Errors, including bad arguments, come
back as tool results that say what went wrong and how to fix the call; nothing
here raises into the agent loop.
"""

from __future__ import annotations

import posixpath
import shlex
import uuid
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError

from agent.harness.sandbox import WORKDIR, ExecResult

MAX_READ_CHARS = 40_000
MAX_LINE_CHARS = 1_000
MAX_BASH_CHARS = 20_000
EDIT_CONTEXT_LINES = 3
FILE_OP_TIMEOUT_S = 60

_DIR_EXIT = 10  # exit code the read command uses to say "this is a directory"


class Shell(Protocol):
    """What the tools need from a sandbox; `Sandbox` provides it."""

    def exec(self, cmd: str, timeout: float = ...) -> ExecResult:
        """Run a shell command in the working directory."""
        ...

    def write_file(self, path: str, content: str) -> None:
        """Write a text file, creating parent directories."""
        ...


class ReadArgs(BaseModel):
    """Arguments for `read`."""

    path: str = Field(
        description="File or directory, relative to the repo root or absolute."
    )
    start_line: int | None = Field(None, description="First line to show, 1-based.")
    end_line: int | None = Field(None, description="Last line to show, inclusive.")


class WriteArgs(BaseModel):
    """Arguments for `write`."""

    path: str = Field(description="File to create or overwrite.")
    content: str = Field(description="The complete new file contents.")


class EditArgs(BaseModel):
    """Arguments for `edit`."""

    path: str = Field(description="File to edit.")
    old: str = Field(
        description="Exact text to replace, including whitespace. Must occur exactly once."
    )
    new: str = Field(description="Replacement text.")


class BashArgs(BaseModel):
    """Arguments for `bash`."""

    command: str = Field(description="Shell command to run.")


class SubmitArgs(BaseModel):
    """`submit` takes no arguments."""


_DESCRIPTIONS = {
    "read": (
        "Show a file with line numbers, or list a directory's entries. Long files "
        "are cut off; pass start_line and end_line to see the rest."
    ),
    "write": "Create a file or overwrite it with new contents.",
    "edit": (
        "Replace one exact occurrence of `old` with `new` in a file. `old` must "
        "match the file character for character and occur exactly once; include "
        "enough surrounding lines to make it unique."
    ),
    "bash": (
        f"Run a bash command in {WORKDIR} and return its exit code and output. "
        "Each call starts a fresh shell in the repo root, so `cd` does not carry "
        "over. There is no network. Long output is cut from the middle. Commands "
        "time out, so run single test files rather than the whole suite."
    ),
    "submit": "Finish: call this once the fix is in place. Your changes are the answer.",
}

_ARGS: dict[str, type[BaseModel]] = {
    "read": ReadArgs,
    "write": WriteArgs,
    "edit": EditArgs,
    "bash": BashArgs,
    "submit": SubmitArgs,
}

TOOL_NAMES = tuple(_ARGS)


class Toolbox:
    """The tools for one run, bound to one sandbox."""

    def __init__(
        self, shell: Shell, *, bash_timeout_s: float, workdir: str = WORKDIR
    ) -> None:
        """Bind the tools to `shell`; relative paths resolve against `workdir`."""
        self._shell = shell
        self._bash_timeout_s = bash_timeout_s
        self._workdir = workdir

    def tools(self) -> list[BaseTool]:
        """LangChain tools, for `bind_tools()`; calling one goes through `run`."""
        return [self._tool(name) for name in TOOL_NAMES]

    def _tool(self, name: str) -> BaseTool:
        def call(**kwargs: Any) -> str:
            return self.run(name, kwargs)

        return StructuredTool.from_function(
            func=call,
            name=name,
            description=_DESCRIPTIONS[name],
            args_schema=_ARGS[name],
        )

    def run(self, name: str, args: dict[str, Any]) -> str:
        """Run tool `name` with the model's raw `args`; always return a string."""
        if name not in _ARGS:
            return f"Error: unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}."
        try:
            parsed = _ARGS[name].model_validate(args)
        except ValidationError as e:
            return f"Error: invalid arguments for {name}: {_brief(e)}"
        try:
            result: str = getattr(self, f"_{name}")(parsed)
        except Exception as e:  # a tool must never crash the loop
            return f"Error: {name} failed: {type(e).__name__}: {e}"
        return result

    def _read(self, args: ReadArgs) -> str:
        path = self._resolve(args.path)
        q = shlex.quote(path)
        result = self._shell.exec(
            f"if [ -d {q} ]; then ls -1Ap -- {q}; exit {_DIR_EXIT}; fi; cat -- {q}",
            timeout=FILE_OP_TIMEOUT_S,
        )
        if result.exit_code == _DIR_EXIT:
            entries = result.output.splitlines()
            return f"{path}/ ({len(entries)} entries)\n" + "\n".join(entries)
        if result.exit_code != 0:
            return f"Error: cannot read {path}: {result.output.strip()}"
        if "\x00" in result.output:
            return f"Error: {path} is a binary file."
        return _numbered(path, result.output, args.start_line, args.end_line)

    def _write(self, args: WriteArgs) -> str:
        path = self._resolve(args.path)
        error = self._put(path, args.content)
        if error:
            return error
        return f"Wrote {path} ({_line_count(args.content)} lines)."

    def _edit(self, args: EditArgs) -> str:
        path = self._resolve(args.path)
        if not args.old:
            return (
                "Error: `old` is empty. To create or replace a whole file, use write."
            )
        if args.old == args.new:
            return "Error: `old` and `new` are the same; nothing to change."
        result = self._shell.exec(
            f"cat -- {shlex.quote(path)}", timeout=FILE_OP_TIMEOUT_S
        )
        if result.exit_code != 0:
            return f"Error: cannot read {path}: {result.output.strip()}"
        text = result.output
        if "�" in text:
            return f"Error: {path} is not valid UTF-8; edit it with bash instead."
        count = text.count(args.old)
        if count == 0:
            return (
                f"Error: `old` was not found in {path}. It must match exactly, "
                "including indentation and line breaks. Read the file again and "
                "copy the text from it."
            )
        if count > 1:
            return (
                f"Error: `old` occurs {count} times in {path}. Include more "
                "surrounding lines so it matches exactly one place."
            )
        start = text.index(args.old)
        new_text = text[:start] + args.new + text[start + len(args.old) :]
        error = self._put(path, new_text)
        if error:
            return error
        first = text.count("\n", 0, start) + 1
        last = first + args.new.count("\n")
        return f"Edited {path}.\n" + _numbered(
            path,
            new_text,
            max(1, first - EDIT_CONTEXT_LINES),
            last + EDIT_CONTEXT_LINES,
            header=False,
        )

    def _bash(self, args: BashArgs) -> str:
        result = self._shell.exec(args.command, timeout=self._bash_timeout_s)
        status = f"exit code: {result.exit_code}"
        if result.timed_out:
            status += f" (timed out after {self._bash_timeout_s:g}s)"
        return f"{status}\n{_truncate_middle(result.output, MAX_BASH_CHARS)}"

    def _submit(self, _args: SubmitArgs) -> str:
        return "Submitted."

    def _resolve(self, path: str) -> str:
        return posixpath.normpath(posixpath.join(self._workdir, path))

    def _put(self, path: str, content: str) -> str | None:
        """Write `content` to `path`, keeping an existing file's mode; return an error."""
        # Copy through a temp file: `cat >` keeps the target's permissions, where
        # writing it directly would reset them and show up as a mode change in the diff.
        tmp = f"/tmp/.agent-write-{uuid.uuid4().hex}"
        self._shell.write_file(tmp, content)
        q, qtmp = shlex.quote(path), shlex.quote(tmp)
        result = self._shell.exec(
            f"mkdir -p -- {shlex.quote(posixpath.dirname(path))} && cat -- {qtmp} > {q}; "
            f"rc=$?; rm -f -- {qtmp}; exit $rc",
            timeout=FILE_OP_TIMEOUT_S,
        )
        if result.exit_code != 0:
            return f"Error: cannot write {path}: {result.output.strip()}"
        return None


def _numbered(
    path: str,
    text: str,
    start_line: int | None,
    end_line: int | None,
    *,
    header: bool = True,
) -> str:
    """Return lines `start_line`..`end_line` with line numbers, capped in length."""
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return f"{path} is empty."
    start = start_line or 1
    end = min(end_line or total, total)
    if start < 1 or start > total:
        return f"Error: start_line {start} is out of range; {path} has {total} lines."
    if end < start:
        return f"Error: end_line {end} is before start_line {start}."

    out: list[str] = []
    size = 0
    shown_end = start - 1
    for n in range(start, end + 1):
        line = lines[n - 1]
        if len(line) > MAX_LINE_CHARS:
            line = (
                line[:MAX_LINE_CHARS] + f"... [{len(line) - MAX_LINE_CHARS} chars cut]"
            )
        row = f"{n:6}\t{line}"
        if size + len(row) > MAX_READ_CHARS and out:
            break
        out.append(row)
        size += len(row) + 1
        shown_end = n
    body = "\n".join(out)
    if not header:
        return body
    note = f"{path} ({total} lines)"
    if shown_end < end:
        note += (
            f"\n[Showing lines {start}-{shown_end}; output capped. "
            f"Use start_line={shown_end + 1} to continue.]"
        )
    return f"{note}\n{body}"


def _truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of `text`, where commands put context and errors."""
    if len(text) <= limit:
        return text
    half = limit // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n\n[... {cut} characters cut from the middle ...]\n\n{text[-half:]}"


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _brief(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'args'}: {err['msg']}"
        for err in error.errors()
    )
