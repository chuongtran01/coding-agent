from agent.harness.dataset import Task
from agent.harness.sandbox import image_for


def _task(instance_id: str) -> Task:
    return Task(
        instance_id=instance_id,
        repo="x/y",
        base_commit="a" * 40,
        problem_statement="",
        FAIL_TO_PASS=("t",),
        PASS_TO_PASS=(),
        gold_patch="",
        test_patch="",
    )


def test_image_for_uses_swebench_naming() -> None:
    assert (
        image_for(_task("pallets__flask-4992"))
        == "swebench/sweb.eval.x86_64.pallets_1776_flask-4992:latest"
    )
    assert (
        image_for(_task("pytest-dev__pytest-5227"))
        == "swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-5227:latest"
    )
