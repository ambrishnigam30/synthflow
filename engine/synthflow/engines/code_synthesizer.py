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
from typing import Union

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ConstraintSet,
    DistributionMap,
    SchemaDefinition,
)
from synthflow.prompts.master_prompt import CODE_SYNTHESIS_RULES
from synthflow.utils.helpers import safe_json_loads

_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are SynthFlow's Glass Box code generator. Generate a standalone Python script that "
    "produces synthetic data. CORE RULES:\n"
    "1. The script MUST define: def generate(row_count: int, seed: int) -> pd.DataFrame\n"
    "2. Use ONLY numpy, pandas, scipy.stats — NO Faker for domain-specific values\n"
    "3. ALL domain knowledge (value pools, distributions) must be embedded as Python constants\n"
    "4. The function must be stateless and deterministic given the same seed\n"
    "5. Return ONLY the Python source code, no explanations, no markdown fences\n"
    "6. Include these imports at the top: import numpy as np; import pandas as pd; "
    "from typing import Optional\n\n"
    "DATA QUALITY MANDATORY RULES:\n"
    + CODE_SYNTHESIS_RULES
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
    "- Causal generation order: {causal_order}\n"
    "- Value pools (embed these as Python constants):\n{value_pools}\n"
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
            raise RuntimeError(
                "Code synthesis failed: LLM provider returned an error. "
                "Please wait 1-2 minutes and retry, or switch to a different provider."
            )

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

        causal_order = knowledge.causal_generation_order or []

        value_pools_info = "\n".join(
            f"  {pool.column_name}: {pool.values[:20]}"
            for pool in knowledge.real_world_value_pools[:10]
        ) if knowledge.real_world_value_pools else "  (no value pools provided)"

        user = _SYNTHESIZER_USER_TEMPLATE.format(
            table_name=table.name,
            columns=columns_info[:15],
            domain=knowledge.domain,
            row_count=row_count,
            seed=seed,
            dist_hints=dist_hints,
            dag_rules=dag_rules_info,
            causal_order=causal_order,
            value_pools=value_pools_info,
        )

        try:
            raw = await self._llm.complete(
                user,
                system_prompt=_SYNTHESIZER_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=4096,
            )
            return self._extract_python_code(raw)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Code synthesis failed: LLM provider returned an error. "
                "Please wait 1-2 minutes and retry, or switch to a different provider."
            ) from exc

    def _extract_python_code(self, raw: str) -> str:
        """Extract Python code from LLM response, clean markdown, validate."""
        # Strip markdown fences
        import re
        code = re.sub(r"```python\s*", "", raw)
        code = re.sub(r"```\s*", "", code).strip()

        # Validate that it's Python with a generate function
        if "def generate" not in code:
            raise RuntimeError(
                "Code synthesis failed: LLM provider returned an error. "
                "Please wait 1-2 minutes and retry, or switch to a different provider."
            )

        # Validate compiles
        try:
            compile(code, "<glass_box>", "exec")
        except SyntaxError as exc:
            raise RuntimeError(
                "Code synthesis failed: LLM provider returned an error. "
                "Please wait 1-2 minutes and retry, or switch to a different provider."
            ) from exc

        return code

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
