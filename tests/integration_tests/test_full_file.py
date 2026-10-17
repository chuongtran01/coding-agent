"""Needs Docker and the first dev task's image (psf__requests-3362)."""

from agent.context import RepoView, get_strategy
from agent.harness.dataset import load_dev_tasks
from agent.harness.sandbox import Sandbox

TASK = load_dev_tasks()[0]
BUDGET = 24_000


def test_full_file_on_a_real_issue() -> None:
    with Sandbox.start(TASK) as sb:
        context = get_strategy("full_file").build(
            TASK.for_agent(), RepoView(sb), BUDGET
        )
    assert 0 < len(context.text) <= BUDGET
    by_path = {item.path: item for item in context.items}
    # requests/models.py is 861 lines, far over the budget: it is trimmed to the
    # definitions the issue names, which keep their real line numbers.
    models = by_path["requests/models.py"]
    assert models.trimmed
    assert "    def iter_content(self, chunk_size=1, decode_unicode=False):" in (
        context.text
    )
