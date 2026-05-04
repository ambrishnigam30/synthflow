# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-008-01 through E-008-06 — GlassBoxCodeSynthesizer tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from synthflow.engines.code_synthesizer import GlassBoxCodeSynthesizer, _fix_known_code_bugs
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
    client.set_response("value constants", _VALID_GENERATE_SCRIPT)
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


# ── E-008-08 through E-008-13: _fix_known_code_bugs brute-force tests ────────

def test_fix_bugs_injects_safe_days_helper() -> None:
    """_fix_known_code_bugs injects _safe_days helper before generate()."""
    code = "import pandas as pd\n\ndef generate(row_count: int, seed: int) -> pd.DataFrame:\n    return pd.DataFrame()\n"
    fixed = _fix_known_code_bugs(code)
    assert "def _safe_days" in fixed
    # Helper must appear BEFORE generate
    assert fixed.index("def _safe_days") < fixed.index("def generate")


def test_fix_bugs_parenthesised_subtraction_dt_days() -> None:
    """(df['end'] - df['start']).dt.days is replaced with _safe_days(...)."""
    code = (
        "import pandas as pd\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    df['days'] = (df['discharge'] - df['admission']).dt.days\n"
        "    return df\n"
    )
    fixed = _fix_known_code_bugs(code)
    assert ".dt.days" not in fixed
    assert "_safe_days(" in fixed


def test_fix_bugs_variable_dt_days() -> None:
    """bare_var.dt.days is replaced with _safe_days(bare_var)."""
    code = (
        "import pandas as pd\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    duration = delta_col.dt.days\n"
        "    return pd.DataFrame({'d': [duration]})\n"
    )
    fixed = _fix_known_code_bugs(code)
    assert "delta_col.dt.days" not in fixed
    assert "_safe_days(delta_col)" in fixed


def test_fix_bugs_to_timedelta_dt_days() -> None:
    """pd.to_timedelta(...).dt.days is also replaced (old bug pattern still caught)."""
    code = (
        "import pandas as pd\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    d = pd.to_timedelta(df['gap'], unit='d').dt.days\n"
        "    return pd.DataFrame({'d': d})\n"
    )
    fixed = _fix_known_code_bugs(code)
    assert ".dt.days" not in fixed


def test_fix_bugs_no_dt_days_unchanged() -> None:
    """Code with no .dt.days is passed through without inserting spurious replacements."""
    code = (
        "import pandas as pd\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    return pd.DataFrame({'a': range(row_count)})\n"
    )
    fixed = _fix_known_code_bugs(code)
    # _safe_days helper is still injected, but no spurious replacements
    assert "_safe_days(" not in fixed.split("def generate")[1]


def test_fix_bugs_date_range_sample() -> None:
    """pd.date_range(...).sample(...) is wrapped in pd.Series(...)."""
    code = (
        "import pandas as pd\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    dates = pd.date_range('2020-01-01', periods=100).sample(row_count)\n"
        "    return pd.DataFrame({'date': dates})\n"
    )
    fixed = _fix_known_code_bugs(code)
    assert "pd.Series(pd.date_range(" in fixed
    assert ").sample(" in fixed


def test_fix_bugs_idempotent() -> None:
    """Applying _fix_known_code_bugs twice does not corrupt the code."""
    code = (
        "import pandas as pd\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    df['days'] = (df['end'] - df['start']).dt.days\n"
        "    return df\n"
    )
    once = _fix_known_code_bugs(code)
    twice = _fix_known_code_bugs(once)
    # Second pass may expand helper again but must not break syntax
    compile(twice, "<test_idempotent>", "exec")


def test_fix_timedelta_days_kwarg() -> None:
    """pd.Timedelta(days=rng.integers(...)) → pd.to_timedelta(..., unit='D')."""
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    rng = np.random.default_rng(seed)\n"
        "    df['tenure'] = pd.Timedelta(days=rng.integers(30, 3650, size=row_count))\n"
        "    return df\n"
    )
    fixed = _fix_known_code_bugs(code)
    assert "pd.Timedelta(days=" not in fixed
    assert "pd.to_timedelta(" in fixed
    assert "unit='D'" in fixed


def test_fix_timedelta_positional_rng() -> None:
    """pd.Timedelta(rng.integers(...)) → pd.to_timedelta(..., unit='D')."""
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def generate(row_count: int, seed: int) -> pd.DataFrame:\n"
        "    rng = np.random.default_rng(seed)\n"
        "    df['age_td'] = pd.Timedelta(rng.integers(0, 365, size=row_count))\n"
        "    return df\n"
    )
    fixed = _fix_known_code_bugs(code)
    assert "pd.Timedelta(rng." not in fixed
    assert "pd.to_timedelta(" in fixed
    assert "unit='D'" in fixed
