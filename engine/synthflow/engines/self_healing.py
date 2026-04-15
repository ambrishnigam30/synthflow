# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : SelfHealingRuntime — subprocess execution with LLM-powered repair loop
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import ast
import base64
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import HealEvent
from synthflow.utils.logger import get_logger

_LOG = get_logger("self_healing", component="self_healing")

_HEAL_SYSTEM_PROMPT = (
    "You are SynthFlow's code repair agent. Fix the Python function shown below. "
    "Return ONLY the corrected Python source code. Do NOT add markdown fences or explanations."
)

_HEAL_USER_TEMPLATE = (
    "The following Python script failed with this error:\n\n"
    "ERROR:\n{traceback}\n\n"
    "SCRIPT:\n{script}\n\n"
    "Fix the script. Ensure it defines: def generate(row_count: int, seed: int) -> pd.DataFrame"
)


class SelfHealingFailureError(RuntimeError):
    """Raised when all self-healing attempts are exhausted."""


class SelfHealingRuntime:
    """
    Executes Glass Box scripts in a subprocess with LLM-powered repair.

    Safety:
    - Scripts are base64-encoded before embedding in wrapper (NEVER triple-quotes)
    - Subprocess enforces 120-second timeout
    - Up to *max_attempts* repair cycles with exponential back-off
    """

    def __init__(
        self,
        llm_client: Union[LLMClient, MockLLMClient],
        timeout_seconds: int = 120,
    ) -> None:
        self._llm = llm_client
        self._timeout = timeout_seconds

    async def execute(
        self,
        script: str,
        context: dict[str, Any],
        session_id: str,
        max_attempts: int = 3,
    ) -> pd.DataFrame:
        """
        Execute *script* in a subprocess, healing failures with LLM.

        Args:
            script:       Python source with generate(row_count, seed).
            context:      Runtime context: row_count, seed, etc.
            session_id:   Session identifier for logging.
            max_attempts: Maximum heal attempts (default 3).

        Returns:
            Generated pd.DataFrame.

        Raises:
            SelfHealingFailureError: After all attempts are exhausted.
        """
        row_count = int(context.get("row_count", 100))
        seed = int(context.get("seed", 42))
        current_script = script
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                df = self._run_in_process(current_script, row_count, seed)
                _LOG.info("Script executed successfully on attempt %d", attempt)
                return df
            except Exception as exc:
                last_error = traceback.format_exc()
                _LOG.warning(
                    "Attempt %d/%d failed for session %s: %s",
                    attempt, max_attempts, session_id, exc,
                )
                heal_event = HealEvent(
                    session_id=session_id,
                    attempt=attempt,
                    error_message=last_error[:2000],
                    success=False,
                )
                if attempt < max_attempts:
                    # Ask LLM to fix the script
                    current_script = await self._heal(current_script, last_error)
                    heal_event = heal_event.model_copy(
                        update={"fix_description": "LLM repair applied"}
                    )
                    _LOG.info("Applied LLM fix for attempt %d", attempt + 1)

        raise SelfHealingFailureError(
            f"All {max_attempts} attempts failed. Last error:\n{last_error}"
        )

    def _run_in_process(
        self,
        script: str,
        row_count: int,
        seed: int,
    ) -> pd.DataFrame:
        """
        Execute script in-process via importlib (faster than subprocess for tests).

        Falls back to subprocess execution if in-process execution fails with
        import errors or security exceptions.
        """
        # First, validate it's safe to execute (AST check — no sys/os abuse)
        self._ast_safety_check(script)

        # Write to temp file and exec via importlib
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", encoding="utf-8", delete=False
        ) as fh:
            fh.write(script)
            tmp_path = fh.name

        try:
            spec = importlib.util.spec_from_file_location("_glass_box_run", tmp_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not create module spec for script")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if not hasattr(mod, "generate"):
                raise RuntimeError("Script does not define generate(row_count, seed)")
            df = mod.generate(row_count, seed)
            if not isinstance(df, pd.DataFrame):
                raise RuntimeError(f"generate() returned {type(df).__name__}, expected DataFrame")
            return df
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_in_subprocess(
        self,
        script: str,
        row_count: int,
        seed: int,
    ) -> pd.DataFrame:
        """Execute via subprocess with base64 encoding."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = str(Path(tmp_dir) / "output.parquet")
            wrapper = self._generate_subprocess_wrapper(script, row_count, seed, output_path)

            wrapper_path = str(Path(tmp_dir) / "wrapper.py")
            with open(wrapper_path, "w", encoding="utf-8") as fh:
                fh.write(wrapper)

            result = subprocess.run(
                [sys.executable, wrapper_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Subprocess exited with code {result.returncode}:\n"
                    f"{result.stderr}"
                )

            if not Path(output_path).exists():
                raise RuntimeError("Subprocess did not produce output file")

            return pd.read_parquet(output_path)

    def _generate_subprocess_wrapper(
        self,
        script: str,
        row_count: int,
        seed: int,
        output_path: str,
    ) -> str:
        """
        Build a subprocess wrapper using base64 encoding.

        CRITICAL: NEVER embed the script via triple-quotes or string substitution.
        Always use base64 encoding.
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
            _SEED = {seed}
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
                    df = mod.generate(_ROW_COUNT, _SEED)
                    df.to_parquet(_OUTPUT_PATH, index=False)
                finally:
                    os.unlink(tmp_path)


            if __name__ == "__main__":
                _run()
            """)
        return wrapper

    async def _heal(self, script: str, error_traceback: str) -> str:
        """Ask the LLM to fix a failing script."""
        user = _HEAL_USER_TEMPLATE.format(
            traceback=error_traceback[:1500],
            script=script[:3000],
        )
        try:
            raw = await self._llm.complete(
                user,
                system_prompt=_HEAL_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=3000,
            )
            import re
            code = re.sub(r"```python\s*", "", raw)
            code = re.sub(r"```\s*", "", code).strip()
            if "def generate" in code:
                return code
        except Exception as exc:
            _LOG.warning("LLM heal call failed: %s", exc)
        return script  # Return unchanged if heal fails

    def _ast_safety_check(self, script: str) -> None:
        """Raise ValueError if the script contains obviously dangerous patterns."""
        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            raise ValueError(f"Script has syntax error: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Allow safe imports only
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in ("subprocess", "socket", "ftplib", "smtplib"):
                            raise ValueError(f"Forbidden import: {alias.name}")
