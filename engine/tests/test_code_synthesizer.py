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


_VALID_GENERATE_SCRIPT = (
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
    "    rng = np.random.default_rng(seed)\n"
    "    return pd.DataFrame({'id': range(row_count), 'value': rng.normal(size=row_count)})\n"
)


def _synth() -> GlassBoxCodeSynthesizer:
    client = MockLLMClient()
    client.set_response("Glass Box", _VALID_GENERATE_SCRIPT)
    return GlassBoxCodeSynthesizer(client)


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


# ── E-008-03: LLM failure raises RuntimeError (no garbage fallback) ──────────

@pytest.mark.asyncio
async def test_code_synthesizer_llm_failure_raises() -> None:
    """When LLM returns a response without def generate, RuntimeError is raised."""
    from synthflow.llm_client import MockLLMClient
    client = MockLLMClient()
    client.set_response("Glass Box", '{"not": "python code"}')
    synth = GlassBoxCodeSynthesizer(client)
    with pytest.raises(RuntimeError, match="Code synthesis failed"):
        await synth.synthesize(
            _minimal_schema(), _knowledge(),
            DistributionMap(), ConstraintSet(),
            row_count=50, seed=42,
        )


# ── E-008-04: extract_python_code validates generate function ────────────

def test_extract_python_code_rejects_missing_generate() -> None:
    """_extract_python_code raises when output has no def generate."""
    synth = _synth()
    with pytest.raises(RuntimeError, match="Code synthesis failed"):
        synth._extract_python_code("x = 1")


# ── E-008-05: extract_python_code rejects syntax errors ──────────────────

def test_extract_python_code_rejects_syntax_error() -> None:
    """_extract_python_code raises on SyntaxError in generated code."""
    synth = _synth()
    bad_code = "def generate(row_count: int, seed: int):\n    return !!invalid"
    with pytest.raises(RuntimeError, match="Code synthesis failed"):
        synth._extract_python_code(bad_code)


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
