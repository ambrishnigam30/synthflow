# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-002-01 through E-002-11 — helpers tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd
import pytest

from synthflow.utils.helpers import (
    LLMParseError,
    compute_prompt_hash,
    detect_timestamp_columns,
    format_number_indian,
    format_number_western,
    generate_session_id,
    safe_json_loads,
)


# ── E-002-01: _at suffix detected ─────────────────────────────────────────

def test_detect_timestamp_columns_matches_created_at() -> None:
    """Column 'created_at' is detected as a timestamp."""
    df = pd.DataFrame({"created_at": ["2024-01-01"], "age": [30]})
    result = detect_timestamp_columns(df)
    assert "created_at" in result
    assert "age" not in result


# ── E-002-02: marks_math NOT detected ─────────────────────────────────────

def test_detect_timestamp_columns_ignores_marks_math() -> None:
    """'marks_math' must NOT be classified as a timestamp column."""
    df = pd.DataFrame(
        {"marks_math": [95], "updated_at": ["2024-06-01"], "category": ["A"]}
    )
    result = detect_timestamp_columns(df)
    assert "marks_math" not in result
    assert "updated_at" in result
    assert "category" not in result


# ── E-002-03: datetime64 dtype detected ───────────────────────────────────

def test_detect_timestamp_columns_matches_datetime_dtype() -> None:
    """Columns with datetime64 dtype are detected regardless of name."""
    df = pd.DataFrame(
        {
            "purchase_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "score": [10, 20],
            "loaded_at": ["2024-01-01", "2024-01-02"],   # _at suffix, string dtype
        }
    )
    result = detect_timestamp_columns(df)
    assert "purchase_date" in result   # datetime64 — no _at suffix but dtype matches
    assert "loaded_at" in result       # _at suffix
    assert "score" not in result


def test_detect_timestamp_columns_empty_dataframe() -> None:
    """Empty DataFrame returns empty list."""
    df = pd.DataFrame()
    assert detect_timestamp_columns(df) == []


def test_detect_timestamp_columns_no_timestamps() -> None:
    """DataFrame with no timestamp-like columns returns empty list."""
    df = pd.DataFrame({"name": ["Alice"], "age": [30], "salary": [50000.0]})
    assert detect_timestamp_columns(df) == []


# ── E-002-04: Indian number format ────────────────────────────────────────

def test_format_number_indian() -> None:
    """1234567.89 → '12,34,567.89'."""
    assert format_number_indian(1234567.89) == "12,34,567.89"


def test_format_number_indian_small() -> None:
    """999.5 → '999.50'."""
    assert format_number_indian(999.5) == "999.50"


def test_format_number_indian_large() -> None:
    """10000000.0 → '1,00,00,000.00' (1 crore)."""
    assert format_number_indian(10_000_000.0) == "1,00,00,000.00"


def test_format_number_indian_negative() -> None:
    """Negative numbers have a leading minus sign."""
    result = format_number_indian(-1234567.89)
    assert result.startswith("-")
    assert "12,34,567.89" in result


# ── E-002-05: Western number format ───────────────────────────────────────

def test_format_number_western() -> None:
    """1234567.89 → '1,234,567.89'."""
    assert format_number_western(1234567.89) == "1,234,567.89"


def test_format_number_western_small() -> None:
    """999.5 → '999.50'."""
    assert format_number_western(999.5) == "999.50"


# ── E-002-06: JSON parse clean ────────────────────────────────────────────

def test_safe_json_loads_parses_clean_json() -> None:
    """Direct JSON string parses correctly."""
    result = safe_json_loads('{"a": 1, "b": "hello"}')
    assert result == {"a": 1, "b": "hello"}


def test_safe_json_loads_parses_json_array() -> None:
    """JSON array parses correctly."""
    result = safe_json_loads('[1, 2, 3]')
    assert result == [1, 2, 3]


# ── E-002-07: markdown fences stripped ────────────────────────────────────

def test_safe_json_loads_strips_json_markdown_fences() -> None:
    """'```json\\n{...}\\n```' parses correctly."""
    text = '```json\n{"key": "value", "num": 42}\n```'
    result = safe_json_loads(text)
    assert result == {"key": "value", "num": 42}


def test_safe_json_loads_strips_generic_markdown_fences() -> None:
    """'```\\n{...}\\n```' (no language tag) parses correctly."""
    text = '```\n{"x": 99}\n```'
    result = safe_json_loads(text)
    assert result == {"x": 99}


# ── E-002-08: preamble extraction ─────────────────────────────────────────

def test_safe_json_loads_extracts_json_from_text() -> None:
    """'Here is the result: {...} Hope this helps' extracts the JSON."""
    text = 'Here is the result: {"domain": "healthcare", "rows": 500} Hope this helps!'
    result = safe_json_loads(text)
    assert result["domain"] == "healthcare"
    assert result["rows"] == 500


# ── E-002-09: garbage raises ──────────────────────────────────────────────

def test_safe_json_loads_raises_on_garbage() -> None:
    """Completely invalid text raises LLMParseError."""
    with pytest.raises(LLMParseError):
        safe_json_loads("hello world, no json here at all!!!")


def test_safe_json_loads_raises_on_empty_string() -> None:
    """Empty string raises LLMParseError."""
    with pytest.raises(LLMParseError):
        safe_json_loads("")


# ── E-002-10: prompt hash deterministic ───────────────────────────────────

def test_compute_prompt_hash_deterministic() -> None:
    """Same prompt always produces the same hash."""
    prompt = "Generate 5000 Indian healthcare patient records"
    h1 = compute_prompt_hash(prompt)
    h2 = compute_prompt_hash(prompt)
    assert h1 == h2
    assert len(h1) == 32  # MD5 hex digest


def test_compute_prompt_hash_different_prompts() -> None:
    """Different prompts produce different hashes."""
    h1 = compute_prompt_hash("prompt A")
    h2 = compute_prompt_hash("prompt B")
    assert h1 != h2


# ── E-002-11: session ID is string ────────────────────────────────────────

def test_generate_session_id_is_string() -> None:
    """generate_session_id() returns a str, not a UUID object."""
    sid = generate_session_id()
    assert isinstance(sid, str)


def test_generate_session_id_unique() -> None:
    """Two calls return different IDs."""
    assert generate_session_id() != generate_session_id()


def test_generate_session_id_uuid_format() -> None:
    """ID has UUID v4 format (36 chars with hyphens)."""
    sid = generate_session_id()
    assert len(sid) == 36
    assert sid.count("-") == 4
