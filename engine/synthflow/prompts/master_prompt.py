# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : master_prompt — World Knowledge Engine system instruction
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

KNOWLEDGE_SYSTEM_INSTRUCTION: str = (
    "You are SynthFlow's World Knowledge Engine: a senior data engineer, domain expert, "
    "statistician, and geographer in one. When given a request, simulate the actual enterprise "
    "system that produces this data in the real world. Think in causality chains. Only use "
    "institutions, names, codes, and values verifiable on the internet. Return valid JSON only."
)

KNOWLEDGE_THINKING_STEPS: str = (
    "Before generating output, think through: "
    "(1) What real enterprise system produces this data? "
    "(2) What is the exact geography — country, state, city, currency, language for names? "
    "(3) What real institutions exist in that specific region? "
    "(4) Build a causal dependency graph — root columns first, derived columns after their parents. "
    "(5) What are realistic distributions for numeric columns? "
    "(6) Provide 20 real verified values per entity column, specific to the requested region — "
    "not generic national values. Names must match gender and region. Names must also reflect birth decade — older people get traditional names, younger people get modern names. "
    "(7) Ensure 12-18 columns, each column should be causally connected to at least 2 others. No filler columns.Every classification column must be derived from its parent column, not generated randomly. The parent determines the child."
)

KNOWLEDGE_ABSOLUTE_RULES: str = (
    "RULES: Return ONLY valid JSON. Never invent unverifiable entities. 12-18 columns. "
    "Every entity column gets 20 real values specific to the requested region. "
    "Name-gender correlation mandatory. "
    "causal_generation_order must list every column once in dependency order."
)

KNOWLEDGE_OUTPUT_SCHEMA: str = (
    "Return JSON with keys: "
    "blueprint_metadata {title, domain, geography {country, state, city, currency_code, timezone}, row_count}, "
    "real_world_value_pools [{pool_name, used_in_column, values[20]}], "
    "column_design [{column_name, data_type, semantic_type, is_primary_key, nullable, "
    "min_value, max_value, enum_values, causal_parents}], "
    "causal_generation_order [list], "
    "temporal_patterns {day_of_week_weights[7], monthly_weights[12]}."
)

CODE_SYNTHESIS_RULES: str = (
    "RULES: "
    "(1) No placeholders — use rng.choice from embedded lists, never name_0. "
    "(2) Numeric values within schema min/max, age as int. "
    "(3) Compute age from DOB or vice versa. "
    "(4) Generate parent columns before children. "
    "(5) Names must match gender and region. "
    "(6) Round monetary values to 2 decimals. "
    "(7) Dates as date type, not datetime. discharge >= admission. "
    "(8) Use int for IDs, never float."
)

KNOWLEDGE_USER_TEMPLATE: str = (
    "Generate a complete CausalKnowledgeBundle for the following request:\n\n"
    "USER PROMPT: {user_prompt}\n"
    "DOMAIN: {domain}\n"
    "REGION: {region}\n"
    "ROW COUNT: {row_count}\n"
    "GEOGRAPHIC CONTEXT FROM DATABASE: {geo_context}\n\n"
    "{thinking_steps}\n\n"
    "{absolute_rules}\n\n"
    "EXPECTED OUTPUT SCHEMA:\n"
    "{output_schema}\n\n"
    "Now generate the complete JSON. Remember: ONLY valid JSON, no markdown, no explanation."
)


def build_knowledge_prompt(
    user_prompt: str,
    domain: str,
    region: str,
    row_count: int,
    geo_context: str,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) tuple for the knowledge LLM call.

    Args:
        user_prompt:  Original natural-language user request.
        domain:       Detected domain (e.g. 'healthcare').
        region:       Region string (e.g. 'India (currency: INR, avg salary USD: 800)').
        row_count:    Number of rows to generate.
        geo_context:  JSON string of DuckDB geo facts.

    Returns:
        (system_prompt, user_message) ready for LLM call.
    """
    system = KNOWLEDGE_SYSTEM_INSTRUCTION
    user = KNOWLEDGE_USER_TEMPLATE.format(
        user_prompt=user_prompt,
        domain=domain,
        region=region,
        row_count=row_count,
        geo_context=geo_context,
        thinking_steps=KNOWLEDGE_THINKING_STEPS,
        absolute_rules=KNOWLEDGE_ABSOLUTE_RULES,
        output_schema=KNOWLEDGE_OUTPUT_SCHEMA,
    )
    return system, user
