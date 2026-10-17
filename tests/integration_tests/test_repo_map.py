"""Needs Docker and the first dev task's image (psf__requests-3362)."""

from agent.context import RepoView, get_strategy
from agent.harness.dataset import load_dev_tasks
from agent.harness.sandbox import Sandbox

TASK = load_dev_tasks()[0]
BUDGET = 24_000


def test_repo_map_on_a_real_issue() -> None:
    with Sandbox.start(TASK) as sb:
        context = get_strategy("repo_map").build(TASK.for_agent(), RepoView(sb), BUDGET)
    assert 0 < len(context.text) <= BUDGET
    assert all(item.kind == "skeleton" for item in context.items)
    # The fix is in requests/utils.py, ranked 7th: too far down for full_file.
    assert "requests/utils.py" in {item.path for item in context.items}
    assert "def iter_content(self, chunk_size=1, decode_unicode=False):" in (
        context.text
    )
