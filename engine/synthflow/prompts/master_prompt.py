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

KNOWLEDGE_THINKING_STEPS: str = """
MANDATORY THINKING PROCESS — complete all 12 steps before generating output:

STEP 1 — DOMAIN: Identify the real enterprise software (e.g., Epic EMR, SAP S/4HANA, Temenos T24), the exact department, and the actual table name it would use in that system.

STEP 2 — GEOGRAPHY: Derive the country/state/city. Then: ISO 4217 currency code, phone format with trunk prefix, postal code format regex, primary language/script for names, and timezone.

STEP 3 — REGULATIONS: List only real applicable laws for this domain and geography (India: DPDP Act 2023, RBI/IRDAI/SEBI; US: HIPAA/SOX/PCI-DSS; EU: GDPR/PSD2). Omit any you are not certain about.

STEP 4 — INSTITUTIONS: List real verifiable organizations (hospitals, banks, companies) grouped as tier_1/tier_2/tier_3. Include only those you are certain exist on the internet.

STEP 5 — CODE STANDARDS: Identify industry codes (ICD-10-CM for healthcare, IFSC for Indian banking, HSN for GST, NPI for US providers). Provide format_regex and 3–5 real example codes.

STEP 6 — CAUSAL DAG: Build the dependency graph. Start with root columns (IDs, gender, date_of_birth). Then derive everything causally: age from date_of_birth; city determines state and postal_code; diagnosis determines treatment; income determines product_tier. List every column with causal_parents and the derivation rule.

STEP 7 — STATISTICS: For every numeric column recall the real distribution: salaries are lognormal; cardiac patient age is truncated normal (mean 62, sd 12, range 18–90); transaction amounts follow a power law; CIBIL scores are normal (mean 720, sd 80, range 300–900). Provide distribution type and parameters.

STEP 8 — TEMPORAL PATTERNS: Derive domain-specific weights summing to 1.0 each: day_of_week_weights (7 values), hour_of_day_weights (24 values), monthly_weights (12 values). Ground these in real patterns (hospital admissions peak Mon–Tue; retail peaks Fri–Sat evening).

STEP 9 — DIRTY DATA: For each column estimate null_rate, typo_rate, and any causal null rule (e.g., patients over 65 have 40% null email_address; unemployed patients have null employer_name).

STEP 10 — VALUE POOLS: Compile exactly 20 real, verified values for every name, place, institution, or entity column. Apply cultural patterns: North Indian male names differ from South Indian; Sikh names use Singh/Kaur suffix; name pools must correlate with gender and region.

STEP 11 — COHERENCE AUDIT: Verify column count is 12–18. Every column connects causally to at least 2 others. No column is both nullable AND a primary key. No two columns carry identical information. Add any missing essential columns for the domain.

STEP 12 — GENERATION ORDER: Topological sort of the causal DAG. Root columns (no parents) come first; derived columns come after ALL their parents. This is the exact code generation order.
"""

KNOWLEDGE_ABSOLUTE_RULES: str = """
ABSOLUTE RULES — any violation causes the output to be rejected:

1. Return ONLY valid JSON. No markdown fences, no explanatory text, no comments.
2. NEVER invent any institution, hospital, bank, person, place, or code not verifiable on the internet. Exclude if uncertain.
3. Column count MUST be 12–18 inclusive.
4. Every numeric column MUST have a distribution_type and real statistical parameters.
5. causal_generation_order MUST list every column exactly once in strict dependency order.
6. real_world_value_pools MUST provide exactly 20 verified values for every entity/name/place column.
7. day_of_week_weights (7), monthly_weights (12), hour_of_day_weights (24) MUST each independently sum to 1.0.
8. Zero logical contradictions: discharge_date after admission_date, age non-negative, delivery_date after purchase_date.
9. Name-gender-region correlation is MANDATORY: male names ONLY from male pool, female names ONLY from female pool.
10. Healthcare patient datasets MUST include at minimum: patient_id, patient_name, age, gender, date_of_birth, diagnosis, admission_date.
"""

KNOWLEDGE_OUTPUT_SCHEMA: str = """
Return JSON with exactly these top-level keys:

{
  "blueprint_metadata": {
    "title": "string",
    "domain": "string",
    "geography": {
      "country": "string", "state": "string", "city": "string",
      "currency_code": "ISO 4217 string", "timezone": "string"
    },
    "regulatory_context": ["list of applicable regulations"],
    "row_count": integer
  },

  "real_world_entities": {
    "institutions": [
      {"real_name": "string", "website": "string", "tier": "tier_1|tier_2|tier_3", "used_in_column": "string"}
    ],
    "code_standards": [
      {"standard_name": "string", "format_regex": "string", "example_real_codes": ["3-5 real codes"], "used_in_column": "string"}
    ]
  },

  "real_world_value_pools": [
    {"pool_name": "string", "used_in_column": "string", "values": ["exactly 20 real verified values"]}
  ],

  "column_design": [
    {
      "column_name": "snake_case_string",
      "data_type": "string|integer|float|boolean|datetime|date|uuid",
      "semantic_type": "e.g. age, salary, diagnosis_code, patient_name",
      "is_primary_key": false,
      "nullable": true,
      "min_value": null,
      "max_value": null,
      "enum_values": [],
      "causal_parents": []
    }
  ],

  "causal_generation_order": ["column_name1", "column_name2"],

  "temporal_patterns": {
    "day_of_week_weights": [7 floats summing to 1.0],
    "monthly_weights": [12 floats summing to 1.0],
    "hour_of_day_weights": [24 floats summing to 1.0]
  },

  "dirty_data_profile": {
    "per_column": [
      {"column_name": "string", "null_rate": 0.0, "typo_rate": 0.0, "causal_null_rule": "string or null"}
    ]
  },

  "dag_rules": [
    {"parent_column": "string", "child_column": "string", "lambda_str": "lambda row, rng: ...", "description": "string"}
  ],

  "currency_code": "ISO 4217 string"
}
"""

CODE_SYNTHESIS_RULES: str = (
    "10 MANDATORY RULES FOR CODE GENERATION:\n"
    "RULE 1 NO PLACEHOLDERS: Never generate values like name_0, category_1, treatment_2. "
    "Use rng.choice(ENTITY_LIST, size=n) where ENTITY_LIST comes from the value pools provided. "
    "Every string column must draw from a realistic list of at least 10 real values. "
    "If no pool is provided, embed realistic domain-appropriate values as constants.\n"
    "RULE 2 NUMERIC CONSTRAINTS: All numeric values must be within the schema min and max. "
    "Use np.clip(generated_values, min_val, max_val). Age must be INTEGER never float — "
    "use .astype(int). Credit scores are integer. Counts are integer.\n"
    "RULE 3 DATE CONSISTENCY: If both date_of_birth and age columns exist, pick date_of_birth as "
    "source of truth and compute age as: age = ((reference_date - dob).dt.days / 365.25).astype(int). "
    "NEVER use pd.date_range with sequential daily intervals — use rng.integers to pick random offsets.\n"
    "RULE 4 CAUSAL IMPLEMENTATION: Generate columns in the causal_generation_order provided. "
    "Generate parent columns first, then condition children on parent values.\n"
    "RULE 5 NAME GENDER GEOGRAPHY CORRELATION: Generate gender first. "
    "Then select names appropriate for that gender AND region AND age bracket using the value pools. "
    "Male names from male pool, female names from female pool. Use np.where or conditional indexing.\n"
    "RULE 6 LOCATION VALUE CORRELATION: If location and economic columns both exist, "
    "higher-tier cities get higher salaries and costs. Use a city-to-salary-multiplier dict.\n"
    "RULE 7 REALISTIC DISTRIBUTIONS: Use the exact distribution type and parameters from the "
    "knowledge bundle. rng.normal(mean, std, size=n) for continuous, "
    "rng.choice(values, p=weights, size=n) for categorical.\n"
    "RULE 8 NULL INJECTION: Follow the dirty_data_profile. Inject nulls causally not randomly. "
    "Example: mask = (df['age'] > 65) & (rng.random(n) < 0.40); df.loc[mask, 'email'] = None\n"
    "RULE 9 TEMPORAL RHYTHMS: Use day_of_week_weights and hour_of_day_weights from knowledge "
    "bundle when generating timestamps. Sample days proportionally.\n"
    "RULE 10 STATE MACHINE: Enforce valid transitions and temporal ordering. "
    "discharge_date must always be after admission_date. "
    "delivery_date must always be after purchase_date. "
    "Compute derived dates as: derived = base_date + pd.to_timedelta(rng.exponential(scale=X, size=n), unit='D')"
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
