# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Structured JSON logger with session_id / phase / component context
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import sys
from typing import Any, MutableMapping, Optional, Tuple


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("session_id", "phase", "component"):
            val = getattr(record, field, None)
            if val is not None:
                entry[field] = val
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            entry["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(entry, ensure_ascii=False)


class _ContextAdapter(logging.LoggerAdapter):
    """Injects context fields (session_id, phase, component) into every record."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(
    name: str,
    *,
    session_id: Optional[str] = None,
    phase: Optional[str] = None,
    component: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger | _ContextAdapter:
    """
    Return a structured JSON logger for the given name.

    If any context keyword is supplied the returned object is a
    _ContextAdapter whose extra fields appear in every emitted record.

    Args:
        name:       Logger namespace (prefixed with 'synthflow.').
        session_id: Optional session UUID to attach to every record.
        phase:      Optional pipeline phase label.
        component:  Optional subsystem name.
        level:      Logging level; default INFO.

    Returns:
        logging.Logger or _ContextAdapter with JSON output.
    """
    logger = logging.getLogger(f"synthflow.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    context: dict[str, Any] = {}
    if session_id is not None:
        context["session_id"] = session_id
    if phase is not None:
        context["phase"] = phase
    if component is not None:
        context["component"] = component

    if context:
        return _ContextAdapter(logger, context)
    return logger
