"""#2053: ``soup eval quant-check`` refuses an unparseable ``--tasks`` file with
exit 3 (usage error), reporting the loader's message *before* any model is
loaded, for every malformed-input case. On ``main`` before the fix the same
inputs escaped as ``Error: ValueError: ...`` with exit 1 after both models had
already been loaded. ``run_quant_check`` also gains an optional pre-parsed
``tasks`` list so the file is parsed once instead of once per model.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

from .conftest import strip_ansi

runner = CliRunner()


def _clean(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output."""
    return " ".join(strip_ansi(output).split())


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    """cwd = tmp_path with two empty dirs standing in for the models."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()


@pytest.fixture()
def forbid_models(monkeypatch):
    """Fail loudly if make_model_generator runs: --tasks must be refused first."""
    calls = []

    def _boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("model load reached; --tasks should have been refused first")

    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", _boom)
    return calls


def _invoke(tasks_name: str) -> object:
    """Invoke ``eval quant-check`` with two stub-model dirs and --allow-stub."""
    return runner.invoke(app, [
        "eval", "quant-check",
        "--before", "before",
        "--after", "after",
        "--tasks", tasks_name,
        "--allow-stub",
    ])


BAD_FILE_CASES = [
    pytest.param(
        lambda td: (td / "tasks.txt", b"hello\n"),
        "Expected .jsonl file, got: .txt",
        id="wrong-extension",
    ),
    pytest.param(
        lambda td: (td / "bad.jsonl", b"not json\n"),
        "Invalid JSON on line 1",
        id="invalid-json",
    ),
    pytest.param(
        lambda td: (td / "bad.jsonl", json.dumps({"expected": "4"}).encode() + b"\n"),
        "missing required field 'prompt'",
        id="missing-prompt",
    ),
    pytest.param(
        lambda td: (td / "empty.jsonl", b""),
        "No eval tasks found in",
        id="empty-jsonl",
    ),
    pytest.param(
        lambda td: (td / "toomany.jsonl", b"\n".join(
            json.dumps({"prompt": f"p{i}", "expected": "a"}).encode()
            for i in range(10_001)
        ) + b"\n"),
        "exceeds maximum of 10000",
        id="over-max-eval-tasks",
    ),
]


@pytest.mark.parametrize(("make_file", "message"), BAD_FILE_CASES)
def test_unparseable_tasks_is_a_usage_error_before_any_model_load(
    cli_env, forbid_models, tmp_path, make_file, message,
):
    """Every malformed --tasks input exits 3 with the loader's message."""
    path, payload = make_file(tmp_path)
    path.write_bytes(payload)

    result = _invoke(path.name)

    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert message in _clean(result.output)
    assert forbid_models == []  # refused before a single model was touched


def test_missing_tasks_file_still_exits_3(cli_env, tmp_path):
    """The up-front existence check keeps its message and exit code."""
    result = _invoke("missing.jsonl")
    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert "--tasks not found: missing.jsonl" in _clean(result.output)


def test_valid_tasks_still_reach_the_scorer(cli_env, tmp_path):
    """A valid file runs end to end on stubs and renders the JSON verdict."""
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps({"prompt": "2+2=", "expected": "4", "scoring": "exact"}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, [
        "eval", "quant-check",
        "--before", "before",
        "--after", "after",
        "--tasks", "tasks.jsonl",
        "--allow-stub",
        "--format", "json",
    ])
    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    assert '"stub": true' in _clean(result.output)


def test_run_quant_check_uses_preparsed_tasks_without_rereading(tmp_path, monkeypatch):
    """run_quant_check accepts the pre-parsed list and never re-opens the file."""
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.eval.quant_check import run_quant_check

    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        '{"prompt": "2+2=", "expected": "4", "scoring": "exact"}\n',
        encoding="utf-8",
    )
    tasks = load_eval_tasks(str(tasks_file))
    tasks_file.unlink()  # gone — scoring must not depend on re-reading it

    def gen(prompt):
        return "4"

    result = run_quant_check(before_gen=gen, after_gen=gen, tasks=tasks)
    row = result.rows[0]
    assert row.before == 1.0 and row.after == 1.0 and row.verdict == "OK"


def test_run_quant_check_keeps_tasks_file_working(tmp_path, monkeypatch):
    """The documented tasks_file path still loads and scores as before."""
    from soup_cli.eval.quant_check import run_quant_check

    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        '{"prompt": "3+3=", "expected": "6", "scoring": "exact"}\n',
        encoding="utf-8",
    )

    def gen(prompt):
        return "6"

    result = run_quant_check(
        before_gen=gen, after_gen=gen, tasks_file=str(tasks_file),
    )
    row = result.rows[0]
    assert row.before == 1.0 and row.after == 1.0 and row.verdict == "OK"
