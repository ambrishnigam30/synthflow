# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for SelfHealingRuntime
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest


_VALID_SCRIPT = """\
import pandas as pd
import numpy as np


def generate(row_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'id': range(row_count),
        'value': rng.normal(size=row_count),
    })
"""

_MULTI_COL_SCRIPT = """\
import pandas as pd
import numpy as np


def generate(row_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'id': range(row_count),
        'name': [f'Record_{i}' for i in range(row_count)],
        'score': rng.uniform(0, 100, size=row_count),
        'category': rng.choice(['A', 'B', 'C'], size=row_count),
    })
"""


# ── Successful execution ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_self_healing_successful_execution(mock_llm_client) -> None:
    """A valid script executes and returns a DataFrame of the correct length."""
    from synthflow.engines.self_healing import SelfHealingRuntime

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    context = {"row_count": 20, "seed": 42}
    result = await runtime.execute(_VALID_SCRIPT, context, "test-session-123")

    assert result is not None
    assert len(result) == 20
    assert "id" in result.columns
    assert "value" in result.columns


@pytest.mark.asyncio
async def test_self_healing_respects_row_count(mock_llm_client) -> None:
    """Generated DataFrame has exactly the requested number of rows."""
    from synthflow.engines.self_healing import SelfHealingRuntime

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    for row_count in [5, 50, 100]:
        context = {"row_count": row_count, "seed": 0}
        result = await runtime.execute(_VALID_SCRIPT, context, f"session-{row_count}")
        assert len(result) == row_count, (
            f"Expected {row_count} rows, got {len(result)}"
        )


@pytest.mark.asyncio
async def test_self_healing_same_seed_same_output(mock_llm_client) -> None:
    """Two executions with the same seed produce identical DataFrames."""
    from synthflow.engines.self_healing import SelfHealingRuntime

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    context = {"row_count": 30, "seed": 7}

    df1 = await runtime.execute(_VALID_SCRIPT, context, "session-seed-a")
    df2 = await runtime.execute(_VALID_SCRIPT, context, "session-seed-b")

    assert list(df1.columns) == list(df2.columns)
    assert len(df1) == len(df2)
    # id column is deterministic
    assert list(df1["id"]) == list(df2["id"])


@pytest.mark.asyncio
async def test_self_healing_multi_column_script(mock_llm_client) -> None:
    """Script returning 4-column DataFrame works correctly."""
    from synthflow.engines.self_healing import SelfHealingRuntime

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    context = {"row_count": 15, "seed": 99}
    result = await runtime.execute(_MULTI_COL_SCRIPT, context, "session-multi")

    assert len(result) == 15
    assert "name" in result.columns
    assert "score" in result.columns
    assert "category" in result.columns


# ── Base64 encoding ────────────────────────────────────────────────────────────

def test_self_healing_uses_base64_encoding(mock_llm_client) -> None:
    """_generate_subprocess_wrapper embeds the script via base64, not triple-quotes."""
    from synthflow.engines.self_healing import SelfHealingRuntime

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    wrapper = runtime._generate_subprocess_wrapper(
        script="import pandas as pd\ndef generate(n, s): return pd.DataFrame({'x': range(n)})",
        row_count=10,
        seed=42,
        output_path="/tmp/test_output.pkl",
    )
    assert "import base64" in wrapper, "Wrapper must import base64"
    assert "base64" in wrapper, "Wrapper must use base64 encoding"
    # Must NOT embed via naive string substitution with triple-quotes that contain the raw script
    assert '"""import pandas' not in wrapper, "Script must not be triple-quote embedded"


def test_self_healing_wrapper_contains_encoded_script(mock_llm_client) -> None:
    """The wrapper contains a base64-decoded execution path."""
    import base64
    from synthflow.engines.self_healing import SelfHealingRuntime

    script = "import pandas as pd\ndef generate(n, s): return pd.DataFrame({'x': range(n)})"
    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    wrapper = runtime._generate_subprocess_wrapper(
        script=script,
        row_count=5,
        seed=1,
        output_path="/tmp/out.parquet",
    )

    expected_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    assert expected_b64 in wrapper, "Wrapper must contain the base64-encoded script"


# ── Failure handling ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_self_healing_fails_after_max_attempts(mock_llm_client) -> None:
    """An un-fixable script raises SelfHealingFailureError after max_attempts=1."""
    from synthflow.engines.self_healing import SelfHealingRuntime, SelfHealingFailureError

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    broken_script = "this is not python code at all @@@ !!!"
    context = {"row_count": 10, "seed": 42}

    with pytest.raises((SelfHealingFailureError, Exception)):
        await runtime.execute(broken_script, context, "test-session-fail", max_attempts=1)


@pytest.mark.asyncio
async def test_self_healing_missing_generate_function_fails(mock_llm_client) -> None:
    """Script that does not define generate() raises an error."""
    from synthflow.engines.self_healing import SelfHealingRuntime, SelfHealingFailureError

    runtime = SelfHealingRuntime(llm_client=mock_llm_client)
    # Valid Python but missing generate()
    no_func_script = "import pandas as pd\nx = 42"
    context = {"row_count": 5, "seed": 0}

    with pytest.raises((SelfHealingFailureError, Exception)):
        await runtime.execute(no_func_script, context, "session-no-func", max_attempts=1)
