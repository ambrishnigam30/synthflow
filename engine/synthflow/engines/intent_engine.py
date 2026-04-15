# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : CognitiveIntentEngine — parses natural-language prompts into IntentObject
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional, Union

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    ImpliedTimeRange,
    IntentObject,
    LocaleInfo,
    RegionInfo,
)
from synthflow.utils.helpers import safe_json_loads
from synthflow.utils.logger import get_logger

_LOG = get_logger("intent_engine", component="intent_engine")

# ── Domain keyword map ─────────────────────────────────────────────────────
# Lists capped at 15 items each to comply with E-014-01 (no inline list > 20 strings)

DOMAIN_KEYWORD_MAP: dict[str, list[str]] = {
    "healthcare": [
        "patient", "hospital", "doctor", "medical", "clinical",
        "health", "diagnosis", "treatment", "pharma", "nurse",
        "physician", "illness", "therapy", "ward", "prescription",
    ],
    "banking": [
        "bank", "loan", "credit", "debit", "account",
        "transaction", "payment", "savings", "deposit", "mortgage",
        "lending", "banking", "investment", "portfolio", "finance",
    ],
    "retail": [
        "store", "shop", "purchase", "order", "customer",
        "product", "inventory", "sku", "sale", "discount",
        "ecommerce", "cart", "checkout", "merchant", "retail",
    ],
    "hr": [
        "employee", "staff", "payroll", "salary", "department",
        "hire", "onboarding", "hr", "human resources", "workforce",
        "attendance", "leave", "appraisal", "headcount", "talent",
    ],
    "agriculture": [
        "crop", "farm", "farmer", "harvest", "soil",
        "irrigation", "yield", "agriculture", "rainfall", "fertilizer",
        "cattle", "livestock", "paddy", "wheat", "sowing",
    ],
    "iot": [
        "sensor", "device", "iot", "telemetry", "reading",
        "temperature", "humidity", "signal", "edge", "firmware",
        "mqtt", "gateway", "actuator", "stream", "timestamp",
    ],
    "insurance": [
        "policy", "premium", "claim", "insured", "beneficiary",
        "coverage", "risk", "underwriting", "insurance", "liability",
        "renewal", "deductible", "agent", "actuarial", "reinsurance",
    ],
    "logistics": [
        "shipment", "delivery", "freight", "warehouse", "tracking",
        "courier", "cargo", "fleet", "route", "logistics",
        "dispatch", "inventory", "supply chain", "fulfillment", "transport",
    ],
    "education": [
        "student", "school", "grade", "teacher", "course",
        "enrollment", "exam", "marks", "curriculum", "university",
        "college", "lecture", "assignment", "academic", "campus",
    ],
    "real_estate": [
        "property", "rent", "lease", "apartment", "house",
        "listing", "mortgage", "realty", "real estate", "tenant",
        "landlord", "broker", "square feet", "valuation", "plot",
    ],
}

# ── Country / region keyword map ───────────────────────────────────────────

_COUNTRY_KEYWORDS: dict[str, str] = {
    "india": "India",
    "indian": "India",
    "usa": "United States",
    "us": "United States",
    "american": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "british": "United Kingdom",
    "england": "United Kingdom",
    "germany": "Germany",
    "german": "Germany",
    "france": "France",
    "french": "France",
    "australia": "Australia",
    "australian": "Australia",
    "canada": "Canada",
    "canadian": "Canada",
    "singapore": "Singapore",
    "brazil": "Brazil",
    "brazilian": "Brazil",
    "china": "China",
    "chinese": "China",
    "japan": "Japan",
    "japanese": "Japan",
    "nigeria": "Nigeria",
    "nigerian": "Nigeria",
    "kenya": "Kenya",
}

_STATE_KEYWORDS: dict[str, tuple[str, str]] = {
    "maharashtra": ("Maharashtra", "India"),
    "mumbai": ("Maharashtra", "India"),
    "karnataka": ("Karnataka", "India"),
    "bangalore": ("Karnataka", "India"),
    "bengaluru": ("Karnataka", "India"),
    "delhi": ("Delhi", "India"),
    "tamil nadu": ("Tamil Nadu", "India"),
    "gujarat": ("Gujarat", "India"),
    "punjab": ("Punjab", "India"),
    "rajasthan": ("Rajasthan", "India"),
    "california": ("California", "United States"),
    "new york": ("New York", "United States"),
    "texas": ("Texas", "United States"),
}

# ── Row count patterns ─────────────────────────────────────────────────────

_ROW_COUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(\d[\d,]*)\s*(?:rows?|records?|entries|samples?|data points?)\b", re.I),
    re.compile(r"\bgenerate\s+(\d[\d,]*)\b", re.I),
    re.compile(r"\bcreate\s+(\d[\d,]*)\b", re.I),
    re.compile(r"\b(\d[\d,]*)\s*(?:rows?|records?)\b", re.I),
]

# ── Stage-1 LLM prompt template ────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = (
    "You are SynthFlow's intent parser. Extract structured information from the user's data "
    "generation prompt. Return ONLY valid JSON matching this schema:\n"
    '{"domain": "string", "sub_domain": "string|null", "row_count": integer, '
    '"country": "string|null", "state_province": "string|null", "city": "string|null", '
    '"economic_tier": "low|lower_middle|upper_middle|high|null", '
    '"scenario": "string|null", "output_format": "csv|json|parquet|excel"}'
)

_INTENT_USER_TEMPLATE = (
    "Parse this data generation request into structured intent JSON:\n\n{prompt}"
)


def _compute_seed(prompt: str) -> int:
    """Deterministic seed: SHA-256(prompt) % 2^31."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31)


def _compute_prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


class CognitiveIntentEngine:
    """
    Converts a free-text prompt into a validated IntentObject.

    Pipeline:
      1. Cache check (7-day TTL via MemoryContextStore)
      2. LLM call with Stage-1 prompt template
      3. Fallback: keyword domain classifier + regex row-count extractor
    """

    def __init__(
        self,
        llm_client: Union[LLMClient, MockLLMClient],
        memory_store: Any = None,
    ) -> None:
        self._llm = llm_client
        self._store = memory_store

    async def parse(self, prompt: str) -> IntentObject:
        """
        Parse *prompt* into an IntentObject.

        Args:
            prompt: Natural-language data generation request.

        Returns:
            Validated IntentObject.
        """
        prompt_hash = _compute_prompt_hash(prompt)

        # 1. Cache check
        if self._store is not None:
            cached = self._store.get_cached_intent(prompt_hash)
            if cached is not None:
                _LOG.info("Intent cache hit for hash %s", prompt_hash)
                return cached

        # 2. Try LLM
        intent = await self._try_llm_parse(prompt)

        # 3. Fallback if LLM didn't produce useful output
        if intent is None:
            intent = self._fallback_parse(prompt)
        else:
            # Ensure seed + row_count are sensible
            intent = self._enrich(intent, prompt)

        # 4. Cache result
        if self._store is not None:
            try:
                self._store.cache_intent(prompt_hash, intent)
            except Exception:
                pass

        return intent

    # ── LLM parsing ───────────────────────────────────────────────────────

    async def _try_llm_parse(self, prompt: str) -> Optional[IntentObject]:
        try:
            raw = await self._llm.complete(
                _INTENT_USER_TEMPLATE.format(prompt=prompt),
                system_prompt=_INTENT_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.3,
                max_tokens=512,
            )
            data = safe_json_loads(raw)
            return self._build_intent_from_dict(data, prompt)
        except Exception as exc:
            _LOG.warning("LLM intent parse failed: %s — using fallback", exc)
            return None

    def _build_intent_from_dict(
        self, data: dict[str, Any], prompt: str
    ) -> Optional[IntentObject]:
        """Try to build an IntentObject from an LLM-returned dict."""
        if not isinstance(data, dict):
            return None
        domain = str(data.get("domain", "")).lower().strip()
        if not domain or domain in ("string", "null", "unknown", "mock"):
            return None

        row_count = int(data.get("row_count", 0) or 0)
        if row_count <= 0:
            row_count = self._extract_row_count(prompt)

        region: Optional[RegionInfo] = None
        country = data.get("country")
        if country and str(country).lower() not in ("null", "none", ""):
            region = RegionInfo(
                country=str(country),
                state_province=data.get("state_province") or None,
                city=data.get("city") or None,
            )
        if region is None:
            region = self._extract_region(prompt)

        sub_domain = data.get("sub_domain")
        if isinstance(sub_domain, str) and sub_domain.lower() in (
            "null", "none", "", "string"
        ):
            sub_domain = None

        return IntentObject(
            domain=domain,
            sub_domain=sub_domain or None,
            region=region,
            row_count=max(1, row_count),
            seed=_compute_seed(prompt),
            output_format=str(data.get("output_format", "csv")),
        )

    # ── Keyword fallback ──────────────────────────────────────────────────

    def _fallback_parse(self, prompt: str) -> IntentObject:
        """
        Build an IntentObject using keyword classification and regex extraction.
        Used when the LLM is unavailable or returns garbage.
        """
        domain = self._keyword_domain(prompt)
        row_count = self._extract_row_count(prompt)
        region = self._extract_region(prompt)
        return IntentObject(
            domain=domain,
            region=region,
            row_count=row_count,
            seed=_compute_seed(prompt),
        )

    def _keyword_domain(self, prompt: str) -> str:
        """Return the domain whose keywords have the most hits in *prompt*."""
        lower = prompt.lower()
        best_domain = "general"
        best_count = 0
        for domain, keywords in DOMAIN_KEYWORD_MAP.items():
            count = sum(1 for kw in keywords if kw in lower)
            if count > best_count:
                best_count = count
                best_domain = domain
        return best_domain

    def _extract_row_count(self, prompt: str) -> int:
        """Extract an integer row count from *prompt* using regex patterns."""
        for pat in _ROW_COUNT_PATTERNS:
            m = pat.search(prompt)
            if m:
                raw = m.group(1).replace(",", "")
                try:
                    val = int(raw)
                    if 1 <= val <= 10_000_000:
                        return val
                except ValueError:
                    continue
        return 1000  # default

    def _extract_region(self, prompt: str) -> Optional[RegionInfo]:
        """Extract country/state from *prompt* using keyword lookup."""
        lower = prompt.lower()

        country: Optional[str] = None
        state: Optional[str] = None

        for kw, state_country in _STATE_KEYWORDS.items():
            if kw in lower:
                state, country = state_country
                break

        if country is None:
            for kw, c in _COUNTRY_KEYWORDS.items():
                # Use word-boundary matching to avoid false positives
                if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                    country = c
                    break

        if country is None:
            return None
        return RegionInfo(country=country, state_province=state)

    def _enrich(self, intent: IntentObject, prompt: str) -> IntentObject:
        """Ensure seed is set; fill in row_count from regex if missing."""
        updates: dict[str, Any] = {}
        if intent.seed is None:
            updates["seed"] = _compute_seed(prompt)
        if intent.row_count == 1000 and self._extract_row_count(prompt) != 1000:
            updates["row_count"] = self._extract_row_count(prompt)
        if not updates:
            return intent
        return intent.model_copy(update=updates)
