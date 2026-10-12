"""The agent's tools: read, write, edit, bash and submit, run inside a sandbox.

Every call returns a `ToolResult`: the text the model sees, plus a few facts for
the harness (error, truncation, exit code, runtime) that are never sent to the
model. Errors, including bad arguments, come back as results that say what went
wrong and how to fix the call; nothing here raises into the agent loop.
"""

from __future__ import annotations

import posixpath
import shlex
import time
import uuid
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError

from agent.harness.sandbox import WORKDIR, ExecResult

# About 4k tokens, or 300-400 lines of code. Every turn resends the history, so a
# whole large file read a few times can use up the token budget (docs/DEV_RUNS.md).
MAX_READ_CHARS = 16_000
MAX_LINE_CHARS = 1_000
MAX_BASH_CHARS = 20_000
EDIT_CONTEXT_LINES = 3
FILE_OP_TIMEOUT_S = 60

_DIR_EXIT = 10  # exit code the read command uses to say "this is a directory"


class ToolResult(BaseModel):
    """The outcome of one tool call."""

    output: str  # exactly what the model sees
    is_error: bool = False  # the tool could not do what was asked
    truncated: bool = False  # output was cut to fit a cap
    exit_code: int | None = None  # bash only
    runtime_s: float = 0.0


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
            return self.run(name, kwargs).output

        return StructuredTool.from_function(
            func=call,
            name=name,
            description=_DESCRIPTIONS[name],
            args_schema=_ARGS[name],
        )

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run tool `name` with the model's raw `args`; never raise."""
        start = time.monotonic()
        result = self._dispatch(name, args)
        return result.model_copy(update={"runtime_s": time.monotonic() - start})

    def _dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in _ARGS:
            return _error(f"unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}.")
        try:
            parsed = _ARGS[name].model_validate(args)
        except ValidationError as e:
            return _error(f"invalid arguments for {name}: {_brief(e)}")
        try:
            result: ToolResult = getattr(self, f"_{name}")(parsed)
        except Exception as e:  # a tool must never crash the loop
            return _error(f"{name} failed: {type(e).__name__}: {e}")
        return result

    def _read(self, args: ReadArgs) -> ToolResult:
        path = self._resolve(args.path)
        q = shlex.quote(path)
        result = self._shell.exec(
            f"if [ -d {q} ]; then ls -1Ap -- {q}; exit {_DIR_EXIT}; fi; cat -- {q}",
            timeout=FILE_OP_TIMEOUT_S,
        )
        if result.exit_code == _DIR_EXIT:
            entries = result.output.splitlines()
            listing = f"{path}/ ({len(entries)} entries)\n" + "\n".join(entries)
            return ToolResult(output=listing)
        if result.exit_code != 0:
            return _error(f"cannot read {path}: {result.output.strip()}")
        if "\x00" in result.output:
            return _error(f"{path} is a binary file.")
        return _numbered(path, result.output, args.start_line, args.end_line)

    def _write(self, args: WriteArgs) -> ToolResult:
        path = self._resolve(args.path)
        error = self._put(path, args.content)
        if error:
            return error
        return ToolResult(output=f"Wrote {path} ({_line_count(args.content)} lines).")

    def _edit(self, args: EditArgs) -> ToolResult:
        path = self._resolve(args.path)
        if not args.old:
            return _error(
                "`old` is empty. To create or replace a whole file, use write."
            )
        if args.old == args.new:
            return _error("`old` and `new` are the same; nothing to change.")
        result = self._shell.exec(
            f"cat -- {shlex.quote(path)}", timeout=FILE_OP_TIMEOUT_S
        )
        if result.exit_code != 0:
            return _error(f"cannot read {path}: {result.output.strip()}")
        text = result.output
        if "\ufffd" in text:
            return _error(f"{path} is not valid UTF-8; edit it with bash instead.")
        count = text.count(args.old)
        if count == 0:
            return _error(
                f"`old` was not found in {path}. It must match exactly, including "
                "indentation and line breaks. Read the file again and copy the text "
                "from it."
            )
        if count > 1:
            return _error(
                f"`old` occurs {count} times in {path}. Include more surrounding "
                "lines so it matches exactly one place."
            )
        start = text.index(args.old)
        new_text = text[:start] + args.new + text[start + len(args.old) :]
        error = self._put(path, new_text)
        if error:
            return error
        first = text.count("\n", 0, start) + 1
        last = first + args.new.count("\n")
        snippet = _numbered(
            path,
            new_text,
            max(1, first - EDIT_CONTEXT_LINES),
            last + EDIT_CONTEXT_LINES,
            header=False,
        )
        return snippet.model_copy(
            update={"output": f"Edited {path}.\n{snippet.output}"}
        )

    def _bash(self, args: BashArgs) -> ToolResult:
        result = self._shell.exec(args.command, timeout=self._bash_timeout_s)
        status = f"exit code: {result.exit_code}"
        if result.timed_out:
            status += f" (timed out after {self._bash_timeout_s:g}s)"
        output = _truncate_middle(result.output, MAX_BASH_CHARS)
        # A non-zero exit is a normal result (a failing test, a reproduced bug);
        # only a command that was killed for running too long counts as an error.
        return ToolResult(
            output=f"{status}\n{output}",
            is_error=result.timed_out,
            truncated=output != result.output,
            exit_code=result.exit_code,
        )

    def _submit(self, _args: SubmitArgs) -> ToolResult:
        return ToolResult(output="Submitted.")

    def _resolve(self, path: str) -> str:
        return posixpath.normpath(posixpath.join(self._workdir, path))

    def _put(self, path: str, content: str) -> ToolResult | None:
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
            return _error(f"cannot write {path}: {result.output.strip()}")
        return None


def _error(message: str) -> ToolResult:
    return ToolResult(output=f"Error: {message}", is_error=True)


def _numbered(
    path: str,
    text: str,
    start_line: int | None,
    end_line: int | None,
    *,
    header: bool = True,
) -> ToolResult:
    """Return lines `start_line`..`end_line` with line numbers, capped in length."""
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return ToolResult(output=f"{path} is empty.")
    start = start_line or 1
    end = min(end_line or total, total)
    if start < 1 or start > total:
        return _error(f"start_line {start} is out of range; {path} has {total} lines.")
    if end < start:
        return _error(f"end_line {end} is before start_line {start}.")

    out: list[str] = []
    size = 0
    shown_end = start - 1
    lines_cut = False
    for n in range(start, end + 1):
        line = lines[n - 1]
        if len(line) > MAX_LINE_CHARS:
            line = (
                line[:MAX_LINE_CHARS] + f"... [{len(line) - MAX_LINE_CHARS} chars cut]"
            )
            lines_cut = True
        row = f"{n:6}\t{line}"
        if size + len(row) > MAX_READ_CHARS and out:
            break
        out.append(row)
        size += len(row) + 1
        shown_end = n
    body = "\n".join(out)
    capped = shown_end < end
    if header:
        note = f"{path} ({total} lines)"
        if capped:
            note += (
                f"\n[Showing lines {start}-{shown_end}; output capped. "
                f"Use start_line={shown_end + 1} to continue.]"
            )
        body = f"{note}\n{body}"
    return ToolResult(output=body, truncated=capped or lines_cut)


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
