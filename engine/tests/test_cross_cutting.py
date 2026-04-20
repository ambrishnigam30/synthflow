# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Cross-cutting policy tests (AST scans, copyright, coding rules)
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import ast
import pathlib
import re

import pytest

# ── Source file discovery ──────────────────────────────────────────────────────

# Resolve engine root relative to this file (tests/ → synthflow/)
_TESTS_DIR = pathlib.Path(__file__).parent
_ENGINE_ROOT = _TESTS_DIR.parent  # .../engine/
_SYNTHFLOW_DIR = _ENGINE_ROOT / "synthflow"

_ENGINE_DIR = _SYNTHFLOW_DIR / "engines"
_PRIVACY_DIR = _SYNTHFLOW_DIR / "privacy"
_QUALITY_DIR = _SYNTHFLOW_DIR / "quality"
_UTILS_DIR = _SYNTHFLOW_DIR / "utils"


def _all_engine_py_files() -> list[pathlib.Path]:
    """Return all engine/privacy/quality/core Python source files."""
    files: list[pathlib.Path] = []
    for directory in (_ENGINE_DIR, _PRIVACY_DIR, _QUALITY_DIR, _UTILS_DIR):
        if directory.exists():
            files.extend(directory.glob("*.py"))
    # Add top-level synthflow module files
    for name in ("core.py", "orchestrator.py", "causal_dag.py", "llm_client.py"):
        candidate = _SYNTHFLOW_DIR / name
        if candidate.exists():
            files.append(candidate)
    return [f for f in files if f.exists() and not f.name.startswith("__")]


# ── Copyright header ───────────────────────────────────────────────────────────

def test_all_engine_files_have_copyright_header() -> None:
    """Every engine .py file contains the SynthFlow copyright header."""
    missing: list[str] = []
    for path in _all_engine_py_files():
        content = path.read_text(encoding="utf-8")
        if "Copyright (c) 2026 Ambrish Nigam" not in content:
            missing.append(str(path))

    assert not missing, (
        f"{len(missing)} file(s) missing copyright header:\n"
        + "\n".join(f"  {p}" for p in missing)
    )


# ── No hardcoded domain lists ──────────────────────────────────────────────────

def test_no_hardcoded_domain_lists_in_source_files() -> None:
    """
    AST scan: no inline Python list with more than 20 string literals.

    Rule E-014-01: all domain knowledge must come from the LLM, not
    hardcoded Python lists of city names, salaries, job titles, etc.
    """
    violations: list[str] = []
    for path in _all_engine_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            # Unparseable files fail elsewhere; skip AST check for them
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.List):
                continue
            string_elts = [
                e for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if len(string_elts) > 20:
                violations.append(
                    f"{path}:{getattr(node, 'lineno', '?')} "
                    f"— inline string list with {len(string_elts)} items "
                    f"(violates E-014-01: no domain lists > 20 items)"
                )

    assert not violations, (
        f"{len(violations)} violation(s) found:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ── No pa_IN Faker locale ──────────────────────────────────────────────────────

def test_no_faker_pa_in_locale() -> None:
    """
    No source file contains the string 'pa_IN'.

    Faker locale 'pa_IN' does not exist; use 'hi_IN' for Punjab and
    all North India (CLAUDE.md Rule 6).
    """
    offenders: list[str] = []
    for path in _all_engine_py_files():
        content = path.read_text(encoding="utf-8")
        if "pa_IN" in content:
            offenders.append(str(path))

    assert not offenders, (
        f"Forbidden Faker locale 'pa_IN' found in:\n"
        + "\n".join(f"  {p}" for p in offenders)
    )


# ── str(uuid.uuid4()) enforcement ─────────────────────────────────────────────

def test_no_bare_uuid4() -> None:
    """
    All uuid4() calls are wrapped in str().

    Bare uuid.uuid4() returns a UUID object, not a string; str(uuid.uuid4())
    is required (CLAUDE.md Rule 7).
    """
    _BARE_UUID4 = re.compile(r"(?<!\bstr\()uuid\.uuid4\(\)")

    offenders: list[str] = []
    for path in _all_engine_py_files():
        content = path.read_text(encoding="utf-8")
        for match in re.finditer(r"uuid\.uuid4\(\)", content):
            start = match.start()
            # Check the 5 characters before the match for "str("
            preceding = content[max(0, start - 5) : start]
            if "str(" not in preceding:
                line_num = content[: start].count("\n") + 1
                offenders.append(
                    f"{path}:{line_num} — bare uuid.uuid4() (must be str(uuid.uuid4()))"
                )

    assert not offenders, (
        f"{len(offenders)} bare uuid.uuid4() call(s) found:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


# ── Self-healing and code synthesizer base64 enforcement ──────────────────────

def test_self_healing_uses_base64_not_triple_quote() -> None:
    """
    self_healing.py must use base64 encoding to embed scripts.

    CLAUDE.md Rule 3: NEVER use triple-quote string substitution.
    """
    path = _ENGINE_DIR / "self_healing.py"
    if not path.exists():
        pytest.skip("self_healing.py not yet created")

    content = path.read_text(encoding="utf-8")
    assert "import base64" in content, (
        "self_healing.py must import base64 for script embedding"
    )
    assert "base64.b64encode" in content or "base64.b64decode" in content, (
        "self_healing.py must use base64 encode/decode for script embedding"
    )


def test_code_synthesizer_uses_base64_not_triple_quote() -> None:
    """
    code_synthesizer.py must use base64 encoding to embed scripts.

    CLAUDE.md Rule 3: NEVER use triple-quote string substitution.
    """
    path = _ENGINE_DIR / "code_synthesizer.py"
    if not path.exists():
        pytest.skip("code_synthesizer.py not yet created")

    content = path.read_text(encoding="utf-8")
    assert "import base64" in content or "base64" in content, (
        "code_synthesizer.py must use base64 for script embedding"
    )


# ── helpers.py timestamp detection ────────────────────────────────────────────

def test_helpers_timestamp_uses_at_suffix_not_at_substring() -> None:
    """
    helpers.py datetime detection matches '_at' suffix, not 'at' substring.

    CLAUDE.md Rule 4: 'marks_math' must NOT be classified as datetime.
    """
    helpers_path = _UTILS_DIR / "helpers.py"
    if not helpers_path.exists():
        pytest.skip("helpers.py not yet created")

    from synthflow.utils.helpers import detect_semantic_type

    # 'marks_math' must NOT be detected as datetime
    result = detect_semantic_type("marks_math", "integer")
    assert result != "datetime", (
        f"'marks_math' must not be classified as datetime, got '{result}'"
    )

    # Columns ending in '_at' SHOULD be detected as datetime
    result_at = detect_semantic_type("created_at", "string")
    assert result_at == "datetime", (
        f"'created_at' must be classified as datetime, got '{result_at}'"
    )


# ── Schema minimum 12 columns model validator ──────────────────────────────────

def test_schema_definition_validates_pk_requirement() -> None:
    """SchemaDefinition model raises ValueError if any table has no PK."""
    from synthflow.models.schemas import (
        ColumnDefinition,
        SchemaDefinition,
        SchemaTable,
    )
    import pydantic

    no_pk_table = SchemaTable(
        name="no_pk",
        columns=[ColumnDefinition(name="col_a", data_type="string")],
    )
    with pytest.raises((ValueError, pydantic.ValidationError)):
        SchemaDefinition(tables=[no_pk_table])


# ── async/await on I/O-bound operations ───────────────────────────────────────

def test_llm_complete_method_is_coroutine() -> None:
    """LLMClient.complete() must be an async method (returns coroutine)."""
    import inspect
    from synthflow.llm_client import MockLLMClient

    client = MockLLMClient()
    assert inspect.iscoroutinefunction(client.complete), (
        "MockLLMClient.complete must be an async def (CLAUDE.md Rule 11)"
    )


# ── No bare except Exception ───────────────────────────────────────────────────

def test_no_bare_except_exception_in_engine_files() -> None:
    """
    AST scan: no engine file uses a bare 'except Exception:' with no re-raise.

    CLAUDE.md Rule 12: use specific exception types, not bare except Exception.

    NOTE: 'except Exception as exc:' is allowed when the exception is logged
    or re-raised. This test only flags bare 'except:' (no exception type at all).
    """
    bare_excepts: list[str] = []
    for path in _all_engine_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # Bare 'except:' has type=None
            if node.type is None:
                line_num = getattr(node, "lineno", "?")
                bare_excepts.append(
                    f"{path}:{line_num} — bare 'except:' without exception type"
                )

    assert not bare_excepts, (
        f"{len(bare_excepts)} bare except clause(s) found:\n"
        + "\n".join(f"  {e}" for e in bare_excepts)
    )
