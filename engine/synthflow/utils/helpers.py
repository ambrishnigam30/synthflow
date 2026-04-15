# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Utility helpers — timestamp detection, number formatting, JSON parsing
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import pandas as pd


# ── Custom exceptions ──────────────────────────────────────────────────────

class LLMParseError(ValueError):
    """Raised when all 6 JSON-extraction strategies fail on an LLM response."""


# ── Timestamp detection ────────────────────────────────────────────────────

def detect_timestamp_columns(df: pd.DataFrame) -> list[str]:
    """
    Return column names that look like timestamps.

    Rules (in priority order):
      1. Column dtype is datetime64 — always included regardless of name.
      2. Column name ends with the literal suffix '_at' — included regardless of dtype.

    Intentionally excluded:
      - Names that merely *contain* 'at' as a substring (e.g. 'marks_math',
        'category', 'latitude') are NOT matched by rule 2.

    Returns:
        Deduplicated list preserving DataFrame column order.
    """
    seen: set[str] = set()
    result: list[str] = []

    for col in df.columns:
        col_str = str(col)
        is_dt_dtype = pd.api.types.is_datetime64_any_dtype(df[col])
        ends_with_at = col_str.endswith("_at")

        if (is_dt_dtype or ends_with_at) and col_str not in seen:
            seen.add(col_str)
            result.append(col_str)

    return result


# ── Number formatting ──────────────────────────────────────────────────────

def format_number_indian(n: float) -> str:
    """
    Format *n* using the Indian numbering system (lakh/crore grouping).

    Example:
        >>> format_number_indian(1234567.89)
        '12,34,567.89'
    """
    negative = n < 0
    abs_str = f"{abs(n):.2f}"
    integer_part, decimal_part = abs_str.split(".")

    if len(integer_part) <= 3:
        formatted_int = integer_part
    else:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups: list[str] = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        groups.reverse()
        formatted_int = ",".join(groups) + "," + last_three

    prefix = "-" if negative else ""
    return f"{prefix}{formatted_int}.{decimal_part}"


def format_number_western(n: float) -> str:
    """
    Format *n* using Western (US/EU) comma grouping.

    Example:
        >>> format_number_western(1234567.89)
        '1,234,567.89'
    """
    return f"{n:,.2f}"


# ── JSON parsing ───────────────────────────────────────────────────────────

def safe_json_loads(text: str) -> Any:
    """
    Extract valid JSON from an LLM response using a 6-strategy fallback chain.

    Strategies (tried in order):
      1. Direct ``json.loads`` on the raw text.
      2. Strip ``\`\`\`json … \`\`\`` fences, then parse.
      3. Strip generic ``\`\`\` … \`\`\`` fences, then parse.
      4. Regex: extract first ``{ … }`` block, then parse.
      5. Regex: extract first ``[ … ]`` block, then parse.
      6. Raise ``LLMParseError``.

    Args:
        text: Raw string from an LLM completion.

    Returns:
        Parsed Python object (dict, list, …).

    Raises:
        LLMParseError: When all 5 extraction attempts fail.
    """
    stripped = text.strip() if text else ""

    # Strategy 1 — direct parse
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 2 — ```json … ``` fences
    m = re.search(r"```json\s*([\s\S]*?)\s*```", stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3 — generic ``` … ``` fences
    m = re.search(r"```\s*([\s\S]*?)\s*```", stripped)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4 — first { … } block
    m = re.search(r"\{[\s\S]*\}", stripped)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 5 — first [ … ] block
    m = re.search(r"\[[\s\S]*\]", stripped)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 6 — give up
    raise LLMParseError(
        f"Could not extract valid JSON after 5 strategies. "
        f"Preview: {stripped[:200]!r}"
    )


# ── Hashing & identity ─────────────────────────────────────────────────────

def compute_prompt_hash(prompt: str) -> str:
    """Return the MD5 hex digest of *prompt*."""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def generate_session_id() -> str:
    """Return a new UUID4 as a plain string (never a bare uuid.UUID object)."""
    return str(uuid.uuid4())
