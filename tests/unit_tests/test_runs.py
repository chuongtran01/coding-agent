import re
from pathlib import Path

from agent.runs import new_run_id, run_dir


def test_new_run_id_is_a_utc_timestamp() -> None:
    assert re.fullmatch(r"\d{8}-\d{6}", new_run_id())


def test_run_dir_groups_tasks_by_run() -> None:
    assert run_dir("r1", "psf__requests-0000", Path("runs")) == Path(
        "runs/r1/psf__requests-0000"
    )
