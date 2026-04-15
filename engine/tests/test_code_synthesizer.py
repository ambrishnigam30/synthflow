# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-008-01 through E-008-06 — GlassBoxCodeSynthesizer tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from synthflow.engines.code_synthesizer import GlassBoxCodeSynthesizer
from synthflow.llm_client import MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    ConstraintSet,
    DistributionMap,
    SchemaDefinition,
    SchemaTable,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _minimal_schema() -> SchemaDefinition:
    cols = [
        ColumnDefinition(name="patient_id", data_type="uuid", is_primary_key=True,
                         unique=True, nullable=False),
        ColumnDefinition(name="name", data_type="string"),
        ColumnDefinition(name="age", data_type="integer", semantic_type="age",
                         min_value=0.0, max_value=120.0),
        ColumnDefinition(name="gender", data_type="string",
                         enum_values=["Male", "Female", "Other"]),
        ColumnDefinition(name="diagnosis", data_type="string"),
    ]
    return SchemaDefinition(tables=[SchemaTable(name="patients", columns=cols)])


def _knowledge() -> CausalKnowledgeBundle:
    return CausalKnowledgeBundle(domain="healthcare")


def _synth() -> GlassBoxCodeSynthesizer:
    return GlassBoxCodeSynthesizer(MockLLMClient())


# ── E-008-01: Generates valid Python ─────────────────────────────────────

@pytest.mark.asyncio
async def test_code_synthesizer_generates_valid_python() -> None:
    """Generated code compiles without SyntaxError."""
    synth = _synth()
    code = await synth.synthesize(
        _minimal_schema(), _knowledge(),
        DistributionMap(), ConstraintSet(),
        row_count=50, seed=42,
    )
    # Must not raise SyntaxError
    compile(code, "<test>", "exec")
    assert len(code) > 50


# ── E-008-02: Code has generate() function ────────────────────────────────

@pytest.mark.asyncio
async def test_code_synthesizer_has_generate_function() -> None:
    """Generated code contains 'def generate'."""
    synth = _synth()
    code = await synth.synthesize(
        _minimal_schema(), _knowledge(),
        DistributionMap(), ConstraintSet(),
        row_count=50, seed=42,
    )
    assert "def generate" in code


# ── E-008-03: Fallback script has no domain-specific Faker calls ──────────

@pytest.mark.asyncio
async def test_code_synthesizer_fallback_no_faker() -> None:
    """Fallback script (used when LLM returns garbage) has no Faker usage."""
    synth = _synth()
    code = synth._fallback_script("patients", [
        {"name": "patient_id", "data_type": "uuid", "is_primary_key": True,
         "enum_values": [], "min_value": None, "max_value": None},
        {"name": "age", "data_type": "integer", "is_primary_key": False,
         "enum_values": [], "min_value": 0, "max_value": 120},
    ], 50, 42)
    assert "Faker" not in code
    assert "faker" not in code.lower()


# ── E-008-04: Code embeds constants ───────────────────────────────────────

@pytest.mark.asyncio
async def test_code_synthesizer_fallback_embeds_enum_constants() -> None:
    """Fallback script embeds enum values as inline constants."""
    synth = _synth()
    code = synth._fallback_script("patients", [
        {"name": "patient_id", "data_type": "uuid", "is_primary_key": True,
         "enum_values": [], "min_value": None, "max_value": None},
        {"name": "status", "data_type": "string", "is_primary_key": False,
         "enum_values": ["active", "inactive", "pending"],
         "min_value": None, "max_value": None},
    ], 50, 42)
    assert "active" in code
    assert "inactive" in code


# ── E-008-05: Code is stateless — same seed → same output ─────────────────

def test_code_synthesizer_fallback_script_is_stateless() -> None:
    """Fallback script returns identical DataFrame for same seed."""
    import importlib
    import sys
    import types
    import uuid

    synth = _synth()
    code = synth._fallback_script("test_table", [
        {"name": "record_id", "data_type": "uuid", "is_primary_key": True,
         "enum_values": [], "min_value": None, "max_value": None},
        {"name": "value", "data_type": "float", "is_primary_key": False,
         "enum_values": [], "min_value": None, "max_value": None},
    ], 10, 99)

    # Execute twice and compare
    mod1 = types.ModuleType("test_glass_1")
    mod1.__dict__["uuid"] = uuid
    exec(compile(code, "<test>", "exec"), mod1.__dict__)
    df1 = mod1.__dict__["generate"](10, 99)

    mod2 = types.ModuleType("test_glass_2")
    mod2.__dict__["uuid"] = uuid
    exec(compile(code, "<test>", "exec"), mod2.__dict__)
    df2 = mod2.__dict__["generate"](10, 99)

    # Numeric columns should be identical
    numeric_cols = df1.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        assert list(df1[col]) == list(df2[col]), f"Column {col} not deterministic"


# ── E-008-06: Uses base64 not triple-quotes in wrapper ────────────────────

def test_code_synthesizer_wrapper_uses_base64() -> None:
    """generate_subprocess_wrapper uses base64 import, not triple-quote embedding."""
    synth = _synth()
    wrapper = synth.generate_subprocess_wrapper(
        "def generate(row_count, seed):\n    import pandas as pd\n    return pd.DataFrame()\n",
        row_count=10,
        output_path="/tmp/test.parquet",
    )
    assert "import base64" in wrapper
    assert "base64.b64decode" in wrapper
    # Must NOT contain the script literal as a triple-quoted string
    assert '"""' not in wrapper or wrapper.count('"""') == 0


def test_code_synthesizer_wrapper_no_triple_quote_embedding() -> None:
    """Wrapper script embeds script as base64 string, never as raw triple-quoted source."""
    synth = _synth()
    script = "def generate(row_count: int, seed: int):\n    import pandas as pd\n    return pd.DataFrame({'a': [1]})\n"
    wrapper = synth.generate_subprocess_wrapper(script, 5, "/tmp/out.parquet")
    # The script content should not appear verbatim (it should be b64-encoded)
    assert "def generate(row_count: int, seed: int):" not in wrapper
    # But base64 machinery should be present
    assert "_SCRIPT_B64" in wrapper
