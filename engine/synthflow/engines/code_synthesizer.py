# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : GlassBoxCodeSynthesizer — generates standalone Python data scripts
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import textwrap
from typing import Any, Optional, Union

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ConstraintSet,
    DistributionMap,
    SchemaDefinition,
)
from synthflow.utils.helpers import safe_json_loads
from synthflow.utils.logger import get_logger

_LOG = get_logger("code_synthesizer", component="code_synthesizer")

_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are SynthFlow's Glass Box code generator. Generate a standalone Python script that "
    "produces synthetic data. RULES:\n"
    "1. The script MUST define: def generate(row_count: int, seed: int) -> pd.DataFrame\n"
    "2. Use ONLY numpy, pandas, scipy.stats — NO Faker for domain-specific values\n"
    "3. ALL domain knowledge (value pools, distributions) must be embedded as Python constants\n"
    "4. The function must be stateless and deterministic given the same seed\n"
    "5. Return ONLY the Python source code, no explanations, no markdown fences\n"
    "6. Include these imports at the top: import numpy as np; import pandas as pd; "
    "from typing import Optional"
)

_SYNTHESIZER_USER_TEMPLATE = (
    "Generate a Glass Box synthetic data script for:\n"
    "- Table: {table_name}\n"
    "- Columns: {columns}\n"
    "- Domain: {domain}\n"
    "- Row count: {row_count}\n"
    "- Seed: {seed}\n"
    "- Distribution hints: {dist_hints}\n"
    "- DAG rules: {dag_rules}\n"
    "Embed all value constants. No Faker imports. Function signature: "
    "def generate(row_count: int, seed: int) -> pd.DataFrame"
)


class GlassBoxCodeSynthesizer:
    """
    Generates standalone Python ``generate(row_count, seed) -> pd.DataFrame`` scripts.

    The generated scripts:
    - Embed all knowledge as Python constants (no API calls, no Faker for domain entities)
    - Are stateless and reproducible given a fixed seed
    - Are safely embedded in subprocess wrappers via base64 encoding
    """

    def __init__(self, llm_client: Union[LLMClient, MockLLMClient]) -> None:
        self._llm = llm_client

    async def synthesize(
        self,
        schema: SchemaDefinition,
        knowledge: CausalKnowledgeBundle,
        distributions: DistributionMap,
        constraints: ConstraintSet,
        row_count: int,
        seed: int,
    ) -> str:
        """
        Generate a standalone Python script.

        Args:
            schema:         Table schema to generate data for.
            knowledge:      Domain knowledge bundle.
            distributions:  Distribution map from StatisticalModelingCore.
            constraints:    Constraint set.
            row_count:      Target row count.
            seed:           Random seed for reproducibility.

        Returns:
            Python source code string.
        """
        table = schema.tables[0] if schema.tables else None
        if table is None:
            return self._minimal_script(row_count, seed)

        columns_info = [
            {"name": c.name, "data_type": c.data_type,
             "semantic_type": c.semantic_type,
             "enum_values": c.enum_values,
             "min_value": c.min_value, "max_value": c.max_value,
             "is_primary_key": c.is_primary_key}
            for c in table.columns
        ]

        dag_rules_info = [
            {"parent": r.parent_column, "child": r.child_column,
             "lambda_str": r.lambda_str[:100]}
            for r in knowledge.dag_rules[:5]
        ]

        dist_hints = {
            col: spec.distribution_type
            for col, spec in list(distributions.column_distributions.items())[:10]
        }

        user = _SYNTHESIZER_USER_TEMPLATE.format(
            table_name=table.name,
            columns=columns_info[:15],
            domain=knowledge.domain,
            row_count=row_count,
            seed=seed,
            dist_hints=dist_hints,
            dag_rules=dag_rules_info,
        )

        try:
            raw = await self._llm.complete(
                user,
                system_prompt=_SYNTHESIZER_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=4096,
            )
            code = self._extract_python_code(raw, table.name, columns_info, row_count, seed)
            return code
        except Exception as exc:
            _LOG.warning("LLM code synthesis failed: %s — using fallback script", exc)
            return self._fallback_script(table.name, columns_info, row_count, seed)

    def _extract_python_code(
        self,
        raw: str,
        table_name: str,
        columns_info: list[dict[str, Any]],
        row_count: int,
        seed: int,
    ) -> str:
        """Extract Python code from LLM response, clean markdown, validate."""
        # Strip markdown fences
        import re
        code = re.sub(r"```python\s*", "", raw)
        code = re.sub(r"```\s*", "", code).strip()

        # Validate that it's Python with a generate function
        if "def generate" not in code:
            _LOG.warning("LLM output missing generate() function — using fallback")
            return self._fallback_script(table_name, columns_info, row_count, seed)

        # Validate compiles
        try:
            compile(code, "<glass_box>", "exec")
        except SyntaxError as exc:
            _LOG.warning("Generated code has SyntaxError: %s — using fallback", exc)
            return self._fallback_script(table_name, columns_info, row_count, seed)

        return code

    def _fallback_script(
        self,
        table_name: str,
        columns_info: list[dict[str, Any]],
        row_count: int,
        seed: int,
    ) -> str:
        """Generate a working fallback script when LLM fails."""
        col_defs: list[str] = []
        for col in columns_info:
            name = col["name"]
            dtype = col.get("data_type", "string")
            enum_vals = col.get("enum_values", [])
            is_pk = col.get("is_primary_key", False)

            if is_pk:
                col_defs.append(f'        "{name}": [str(uuid.uuid4()) for _ in range(n)],')
            elif enum_vals and isinstance(enum_vals, list) and len(enum_vals) > 0:
                safe_vals = [repr(v) for v in enum_vals[:20]]
                vals_str = "[" + ", ".join(safe_vals) + "]"
                col_defs.append(
                    f'        "{name}": rng.choice({vals_str}, size=n).tolist(),'
                )
            elif dtype == "integer":
                col_defs.append(f'        "{name}": rng.integers(1, 1000, size=n).tolist(),')
            elif dtype == "float":
                col_defs.append(f'        "{name}": rng.normal(loc=50.0, scale=15.0, size=n).tolist(),')
            elif dtype == "boolean":
                col_defs.append(f'        "{name}": rng.integers(0, 2, size=n).astype(bool).tolist(),')
            elif dtype in ("datetime", "date"):
                col_defs.append(
                    f'        "{name}": pd.date_range("2020-01-01", periods=n, freq="D").tolist(),'
                )
            else:
                col_defs.append(f'        "{name}": [f"{name}_{{i}}" for i in range(n)],')

        col_block = "\n".join(col_defs)

        script = textwrap.dedent(f"""\
            import uuid
            import numpy as np
            import pandas as pd
            from typing import Optional


            def generate(row_count: int, seed: int) -> pd.DataFrame:
                \"\"\"
                Glass Box generated function for table: {table_name}
                Stateless — same seed always produces same DataFrame.
                \"\"\"
                n = row_count
                rng = np.random.default_rng(seed)
                data = {{
            {col_block}
                }}
                return pd.DataFrame(data)
            """)
        return script

    def _minimal_script(self, row_count: int, seed: int) -> str:
        """Absolute minimal fallback when schema is unavailable."""
        return textwrap.dedent(f"""\
            import numpy as np
            import pandas as pd
            from typing import Optional


            def generate(row_count: int, seed: int) -> pd.DataFrame:
                rng = np.random.default_rng(seed)
                return pd.DataFrame({{
                    "id": range(row_count),
                    "value": rng.normal(size=row_count),
                }})
            """)

    def generate_subprocess_wrapper(
        self, script: str, row_count: int, output_path: str
    ) -> str:
        """
        Wrap *script* in a subprocess-safe runner using base64 encoding.

        The generated wrapper:
        - base64-encodes the script (NEVER triple-quote embeds it)
        - Decodes and writes to a temp file
        - Calls generate(row_count, seed) and saves result as Parquet

        Args:
            script:      Python source to wrap.
            row_count:   Number of rows to generate.
            output_path: Path where the output Parquet should be saved.

        Returns:
            Python source string for the wrapper script.
        """
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")

        wrapper = textwrap.dedent(f"""\
            import base64
            import importlib.util
            import os
            import sys
            import tempfile

            import pandas as pd

            _SCRIPT_B64 = "{encoded}"
            _ROW_COUNT = {row_count}
            _OUTPUT_PATH = {repr(output_path)}


            def _run() -> None:
                source = base64.b64decode(_SCRIPT_B64).decode("utf-8")
                with tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w", encoding="utf-8", delete=False
                ) as fh:
                    fh.write(source)
                    tmp_path = fh.name
                try:
                    spec = importlib.util.spec_from_file_location("_glass_box", tmp_path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    df = mod.generate(_ROW_COUNT, 42)
                    df.to_parquet(_OUTPUT_PATH, index=False)
                finally:
                    os.unlink(tmp_path)


            if __name__ == "__main__":
                _run()
            """)
        return wrapper
