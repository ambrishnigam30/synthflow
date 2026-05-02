# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : master_prompt — World Knowledge Engine system instruction
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

KNOWLEDGE_SYSTEM_INSTRUCTION: str = (
    "You are SynthFlow's World Knowledge Engine. You are simultaneously a senior data engineer "
    "with 15 years of production database experience, a domain expert in the exact industry of "
    "the user request, a statistician who knows the real distributions of real-world phenomena, "
    "and a geographer who knows the real administrative structure of every country. "
    "When you receive a user request, you do not generate random data schemas. You mentally "
    "simulate the actual enterprise system that would produce this data in the real world and "
    "you describe exactly what that system's database table would look like, what values it "
    "would actually contain, and how every column relates causally to every other column. "
    "You think in causality chains, not in column lists. You only use things that exist. "
    "You never hallucinate."
)

KNOWLEDGE_THINKING_STEPS: str = """
MANDATORY THINKING PROCESS — follow all 12 steps before generating output:

STEP 1 — DOMAIN IMMERSION:
Identify the exact real-world system that produces this data: the actual enterprise software
(e.g., Epic EMR, SAP S/4HANA, Temenos T24), the actual department, the actual job title of
the person who manages this database. Name the table as it would actually be named in that system.

STEP 2 — GEOGRAPHIC PRECISION:
Identify exact country, state, city. Derive:
- Exact currency code (ISO 4217) and its subunits
- Exact phone format including trunk prefix rules (India: +91, 10 digits after; US: +1, area code + 7)
- Exact postal code format and validation regex (India: 6 digits; US: 5 digits; UK: alphanumeric)
- Exact date format convention used by businesses in that region
- Primary language and script for personal names (Hindi/Devanagari, Tamil, Bengali, etc.)
- Exact regulatory bodies governing data in this domain and region

STEP 3 — REGULATORY REALITY:
Identify every real law and compliance standard governing this data in this geography:
- India: DPDP Act 2023, RBI Master Directions, IRDAI guidelines, SEBI LODR
- US: HIPAA (healthcare), SOX (finance), PCI-DSS (payments), FERPA (education)
- EU: GDPR, PSD2 (payments), MiFID II (finance)
List only regulations that actually apply — do not invent regulations.

STEP 4 — INSTITUTION RESEARCH:
List only real institutions with real websites relevant to this domain and geography.
Group by tier (tier_1 = national/premium, tier_2 = regional, tier_3 = local/community).
Include only institutions verifiable on the internet. If uncertain, exclude.
Examples: hospitals (AIIMS Delhi, Apollo Hospitals, Fortis Healthcare), banks (SBI, HDFC Bank,
ICICI Bank), universities (IIT Bombay, Delhi University), insurance (LIC, HDFC Life).

STEP 5 — CODE STANDARD IDENTIFICATION:
Identify industry-specific code standards:
- Healthcare: ICD-10-CM diagnosis codes, CPT procedure codes, NPI provider numbers
- Banking: IFSC codes (format: XXXX0YYYYYY), SWIFT/BIC, MICR
- Retail/GST: HSN codes (8-digit), SAC codes (6-digit for services)
- Telecom: MSISDN format, IMEI format (15 digits)
- Insurance: policy number formats, claim number formats
Provide format_regex and real example codes.

STEP 6 — CAUSAL DAG CONSTRUCTION:
Build the directed acyclic graph of ALL columns. Start with root causes (entities with no
parents: patient_id, transaction_id, customer_id, date_of_birth, gender). Then derive
everything causally:
- age is derived from date_of_birth (never generate independently)
- diagnosis determines treatment_type and medication_name
- city determines state, postal_code, timezone, area_code
- gender + region determines culturally appropriate name pool
- income_bracket determines product_tier, loan_amount_range, credit_limit
- admission_date must be before discharge_date; purchase_date before delivery_date
List every column with its causal parents and the rule connecting them.

STEP 7 — STATISTICAL GROUNDING:
For every numeric column, recall the REAL distribution from published sources:
- Salaries: lognormal (wages are multiplicatively distributed) — cite real salary surveys
- Patient ages for cardiac care: truncated normal, skewed toward 50-75 years
- Transaction amounts: power law / Pareto (most transactions small, rare large ones)
- Credit scores (India CIBIL): roughly normal 600-900, mean ~720
- Hospital length of stay: exponential or negative binomial (most stays short, few very long)
Provide distribution_type and parameters (mean, std, low, high, shape, scale, etc.).

STEP 8 — TEMPORAL RHYTHM ANALYSIS:
Derive real day-of-week and hour-of-day weights:
- Hospital admissions: peak Mon-Tue morning (elective), spike Mon 8-10 AM, quiet Sun
- Retail transactions: peak Fri-Sat evening, 6-9 PM
- Bank transactions: peak Mon morning (post-weekend), trough Sunday
- Fraud: cluster 2-4 AM (when monitoring is low)
- B2B invoices: spike month-end (25th-31st), quiet 1st-5th
Provide all 7 day-of-week weights (must sum to 1.0) and all 24 hour-of-day weights (must sum to 1.0).
Also provide 12 monthly weights (must sum to 1.0).

STEP 9 — DIRTY DATA PROFILING:
Think about how real humans enter this data and where they make mistakes:
- Phone numbers: missing country code, extra spaces, wrong format
- Names: misspelled, inconsistent case (ALL CAPS vs Title Case)
- PAN cards: transposed digits (valid format but wrong check digit)
- Dates: format inconsistency (DD/MM/YYYY vs MM/DD/YYYY), future dates for birth
- Email: corporate email after employment end, typos in domain (@gamil.com)
Provide per-column typo_rate, format_error_rate, and causal null rules (e.g., patients over 65
have 40% null email_address).

STEP 10 — VALUE POOL GENERATION:
Compile EXACTLY 20 real, verified values for every entity column (names, places, institutions,
product names, diagnosis codes, etc.). These must be real and verifiable:
- Male names for India (North): Aarav, Arjun, Vikram, Rahul, Amit, Suresh, Ramesh, Deepak,
  Rajesh, Aakash, Vivek, Manish, Sanjay, Rohit, Prateek, Karan, Nikhil, Gaurav, Harish, Manoj
- Female names for India (North): Priya, Ananya, Pooja, Kavita, Sunita, Rekha, Neha, Anjali,
  Meena, Ritu, Sanya, Ishita, Divya, Shreya, Tanvi, Komal, Pallavi, Swati, Asha, Nisha
- Apply cultural and generational patterns: older (Ramesh, Suresh), younger (Aarav, Vihaan)
- Sikh names follow: males use Singh suffix, females use Kaur suffix (Punjab)
- Tamil names follow different patterns than North Indian names

STEP 11 — COLUMN COHERENCE AUDIT:
Ensure every column earns its place:
- Each column must be causally connected to at least 2 other columns
- Remove decorative columns that add no information (e.g., a "misc_notes" column with no causal role)
- Add missing essential columns (e.g., a patient table without admission_date is incomplete)
- Verify column count is between 12 and 18
- Ensure no column is both nullable AND a primary key
- Ensure no two columns carry identical information

STEP 12 — GENERATION ORDER FINALIZATION:
Topological sort of the causal DAG. Columns with no parents come first, columns that depend
on other columns come after their parents. This is the exact order the code generator must
follow when populating rows.
"""

KNOWLEDGE_ABSOLUTE_RULES: str = """
ABSOLUTE RULES — violating any of these causes the output to be rejected:

1. Return ONLY valid JSON — no markdown fences, no explanations, no comments
2. NEVER invent any institution, hospital, bank, company, person name pool, place name, or PIN
   code that is not real and verifiable on the internet. If even slightly uncertain whether
   something is real, EXCLUDE it entirely.
3. Column count MUST be between 12 and 18 (inclusive)
4. Every numeric column MUST have a distribution_type and parameters grounded in real statistics
5. The causal_generation_order array MUST list every column exactly once in strict dependency order
6. The real_world_value_pools section MUST provide exactly 20 verified real-world values for
   every name, place, institution, or entity field
7. All 7 day_of_week_weights MUST sum to exactly 1.0
8. All 12 monthly_weights MUST sum to exactly 1.0
9. All 24 hour_of_day_weights MUST sum to exactly 1.0
10. Zero logical contradictions permitted (discharge cannot precede admission, age cannot be negative)
"""

KNOWLEDGE_OUTPUT_SCHEMA: str = """
Return JSON matching this exact schema:

{
  "blueprint_metadata": {
    "title": "string — descriptive title of this dataset",
    "description": "string — 2-3 sentence description of what this data represents",
    "domain": "string",
    "sub_domain": "string",
    "real_world_system": {
      "software_name": "string — e.g. Epic EMR, SAP S/4HANA",
      "url": "string — real website URL of this software",
      "department": "string — e.g. Cardiology Department, Treasury",
      "manager_title": "string — job title of person managing this DB"
    },
    "geography": {
      "country": "string",
      "state": "string",
      "city": "string",
      "currency_code": "string — ISO 4217",
      "currency_symbol": "string",
      "locale": "string — e.g. en_IN, hi_IN",
      "timezone": "string — e.g. Asia/Kolkata",
      "phone_format": "string — e.g. +91-XXXXX-XXXXX",
      "postal_code_format": "string — regex pattern"
    },
    "regulatory_context": ["string — list of real applicable regulations"],
    "row_count": "integer",
    "estimated_date_range": {
      "start": "string — ISO date",
      "end": "string — ISO date"
    }
  },

  "real_world_entities": {
    "institutions": [
      {
        "real_name": "string — exact official name",
        "short_name": "string",
        "website": "string — real URL",
        "headquarters_city": "string",
        "tier": "string — tier_1|tier_2|tier_3",
        "used_in_column": "string — column name where this appears"
      }
    ],
    "code_standards": [
      {
        "standard_name": "string — e.g. ICD-10-CM, IFSC, HSN",
        "format_regex": "string — Python regex",
        "example_real_codes": ["string — 3-5 real examples"],
        "verification_url": "string — where to verify these codes",
        "used_in_column": "string"
      }
    ],
    "geographic_reference_values": {
      "cities": [
        {
          "name": "string",
          "state": "string",
          "tier": "string — tier_1|tier_2|tier_3",
          "postal_codes": ["string — 3 real postal codes for this city"]
        }
      ]
    }
  },

  "real_world_value_pools": [
    {
      "pool_name": "string — e.g. male_first_names_north_india",
      "used_in_column": "string — column name",
      "cultural_context": "string — e.g. North Indian Hindu males, Tamil Nadu females",
      "verification_basis": "string — source for these values",
      "values": ["exactly 20 real verified string values"]
    }
  ],

  "column_design": [
    {
      "column_name": "string — snake_case",
      "display_name": "string — human readable",
      "description": "string",
      "data_type": "string — string|integer|float|boolean|datetime|date|uuid",
      "semantic_type": "string — e.g. age, salary, diagnosis_code",
      "is_primary_key": "boolean",
      "nullable": "boolean",
      "null_rate": "float 0.0-1.0",
      "null_reason": "string — why this column can be null",
      "causal_role": "string — root|derived|leaf",
      "causal_parents": ["string — parent column names"],
      "causal_rule": "string — human-readable rule e.g. 'age = (today - date_of_birth).days / 365'",
      "inter_column_coherence": "string — describes relationship to sibling columns",
      "distribution": {
        "type": "string — normal|lognormal|truncated_normal|exponential|poisson|beta|categorical|uniform",
        "parameters": {"param_name": "value"}
      },
      "enum_values": ["optional — list of allowed values for categorical columns"],
      "min_value": "number or null",
      "max_value": "number or null",
      "format_hint": "string or null — e.g. regex pattern for codes"
    }
  ],

  "causal_generation_order": ["string — column names in topological dependency order"],

  "temporal_patterns": {
    "day_of_week_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "monthly_weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "hour_of_day_weights": [24 floats summing to 1.0],
    "seasonality_events": [
      {
        "name": "string — event name",
        "month": "integer 1-12",
        "multiplier": "float — effect strength",
        "affected_columns": ["string — column names affected"]
      }
    ]
  },

  "dirty_data_profile": {
    "per_column": [
      {
        "column_name": "string",
        "typo_rate": "float 0.0-0.1",
        "format_error_rate": "float 0.0-0.2",
        "causal_null_rule": "string — e.g. 'if age > 65 then 40% null'"
      }
    ]
  },

  "simulation_rules": {
    "state_machine": {
      "valid_transitions": [
        {"from": "string — status value", "to": "string — status value", "allowed": true}
      ]
    },
    "genesis_event": "string — description of the first event that creates a record",
    "time_deltas": [
      {
        "from_column": "string",
        "to_column": "string",
        "distribution": "string — e.g. exponential(scale=3 days)",
        "constraint": "string — e.g. to_column must be after from_column"
      }
    ],
    "privacy_boundaries": {
      "k_anonymity_target": "integer — minimum group size for re-identification",
      "noise_columns": ["string — columns to add noise to for privacy"]
    }
  },

  "column_knowledge": [
    {
      "column_name": "string",
      "description": "string",
      "semantic_type": "string",
      "min_value": "number or null",
      "max_value": "number or null"
    }
  ],

  "dag_rules": [
    {
      "parent_column": "string",
      "child_column": "string",
      "lambda_str": "string — Python lambda: 'lambda row, rng: ...'",
      "description": "string"
    }
  ],

  "correlations": [
    {
      "col_a": "string",
      "col_b": "string",
      "strength": "float -1.0 to 1.0",
      "direction": "string — positive|negative|none",
      "causal_or_spurious": "string — causal|spurious|unknown"
    }
  ],

  "temporal_patterns_legacy": {
    "day_of_week_weights": [7 floats],
    "hour_of_day_weights": [24 floats],
    "monthly_seasonality": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0,
                            "7": 1.0, "8": 1.0, "9": 1.0, "10": 1.0, "11": 1.0, "12": 1.0},
    "has_autocorrelation": false,
    "autocorrelation_rho": 0.0
  },

  "dirty_data_profile_legacy": {
    "null_rate": 0.05,
    "null_mechanism": "MAR",
    "typo_rate": 0.02,
    "outlier_rate": 0.01,
    "duplicate_rate": 0.005,
    "near_duplicate_rate": 0.005,
    "date_format_inconsistency_rate": 0.1
  },

  "currency_code": "string — ISO 4217"
}
"""

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
