# CLAUDE.md — SynthFlow v2.0 Development Instructions

This file is the instruction manual for Claude Code working on the SynthFlow project.
Read this COMPLETELY before writing any code.

---

## Project Overview

SynthFlow is a commercial SaaS platform for autonomous synthetic data generation.
Monorepo with three packages: frontend (Next.js), backend (FastAPI), engine (Python).

**Customer-facing tagline**: "Data that understands the real world"
**Logo**: SynthFlow "S" mark with upward arrow. Dark charcoal (#3d4043). Logo file at /public/logo.svg
**Logo tagline**: "INTEGRATED SYNERGY" (appears below logo mark)

---

## Repository Structure

```
synthflow/
├── frontend/          Next.js 14 (App Router) + TypeScript + Tailwind
├── backend/           FastAPI + SQLAlchemy + Alembic
├── engine/            SynthFlow core engine (15 subsystems)
│   └── synthflow/
│       ├── models/    Pydantic v2 data models
│       ├── utils/     Logger, helpers
│       ├── engines/   All 15 subsystems
│       ├── privacy/   Presidio guard
│       ├── quality/   Quality reporter
│       ├── core.py    DI container
│       └── orchestrator.py  9-phase pipeline
├── docs/              PRD, architecture, API docs
└── docker-compose.yml
```

---

## Absolute Rules — Never Violate

### Engine Rules
1. ZERO hardcoded domain data. No Python lists of city names, school names, salary ranges, job titles, or any domain-specific values. The LLM generates all domain knowledge.
2. Every engine file starts with the copyright header:
   ```python
   # ───────────────────────────────────────────────────────────────
   # Copyright (c) 2026 Ambrish Nigam
   # Author : Ambrish Nigam | https://github.com/ambrishnigam30
   # Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
   # License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
   # Module : [one-line description]
   # ───────────────────────────────────────────────────────────────
   ```
3. self_healing.py and code_synthesizer.py MUST use base64 encoding for script embedding. NEVER use triple-quote string substitution.
4. helpers.py timestamp detection MUST match `_at` suffix, NOT `at` substring. `marks_math` must NOT be classified as datetime.
5. privacy score: k_anonymity=1 means every row is unique = perfect privacy for synthetic data. Score it 40/40, not 0/40.
6. NEVER use Faker locale `pa_IN` (does not exist). Use `hi_IN` for Punjab and all North India.
7. Always `str(uuid.uuid4())`, never bare `uuid.uuid4()`.
8. Schema generation must enforce minimum 12 columns.
9. Generated Glass Box code must be stateless: `generate(row_count, seed) -> pd.DataFrame` is a pure function.
10. Type hints on ALL function signatures (input AND output).
11. async/await on all I/O-bound operations (LLM calls, file I/O, DB queries).
12. No bare `except Exception`. Use specific exception types.

### Backend Rules
1. API keys encrypted AES-256 before database storage.
2. JWT access tokens: 15 min expiry. Refresh tokens: 7 days.
3. All endpoints return consistent JSON: `{"data": ..., "error": null}` or `{"data": null, "error": {"message": "...", "code": "..."}}`
4. Rate limiting per user per plan tier.
5. CORS: frontend domain only.
6. No stack traces in API responses. Log them server-side.

### Frontend Rules (General)
1. TypeScript strict mode. No `any` types.
2. All API calls via React Query with proper loading/error/empty states.
3. No competitor mentions (MOSTLY AI) on any customer-facing page.
4. The SynthFlow logo color is dark charcoal (#3d4043).

---

## TWO DESIGN SYSTEMS — READ CAREFULLY

The frontend uses TWO distinct design systems:

**MARKETING PAGES** (Stripe-inspired): Landing page, pricing, login, signup, docs, blog — everything a visitor sees BEFORE logging in. Premium, financial-grade trust.

**APP PAGES** (Cursor-inspired): Dashboard, generate, explore, history, settings — everything a logged-in user works with daily. Warm, tool-focused, comfortable for extended use.

**What stays consistent across both:**
- SynthFlow logo (dark charcoal #3d4043)
- The brand accent concept (purple in marketing, orange in app)
- Professional, warm tone (never cold/clinical)
- Conservative border-radius (4px-8px, nothing pill-shaped on marketing; 8px standard on app)

The transition point is LOGIN. User mentally switches from "browsing" to "working."

---

## DESIGN SYSTEM A: MARKETING PAGES (Stripe-Inspired)

**Applies to**: `/` (landing), `/pricing`, `/login`, `/signup`, `/docs`, any public page

### Font Families (open-source equivalents)
```
Primary (all text):  "Geist", system-ui, -apple-system, sans-serif
Monospace (code):    "Source Code Pro", ui-monospace, monospace
```
Load Geist from Vercel's @next/font/google or fontsource. It is the closest open-source match to Söhne.
Enable OpenType features: `font-feature-settings: "ss01"` on all Geist text for geometric alternate glyphs.

### Color Palette

#### Primary
```css
--m-purple:           #533afd;   /* Primary brand, CTAs, links */
--m-purple-hover:     #4434d4;   /* Purple hover state */
--m-purple-deep:      #2e2b8c;   /* Icon hover, deep accents */
--m-purple-light:     #b9b9f9;   /* Subdued hover backgrounds */
--m-navy:             #061b31;   /* Heading text — NOT black */
--m-white:            #ffffff;   /* Page background, card surfaces */
--m-brand-dark:       #1c1e54;   /* Dark sections, footer */
--m-dark-navy:        #0d253d;   /* Darkest neutral */
```

#### Body & Labels
```css
--m-body:             #64748d;   /* Secondary text, descriptions */
--m-label:            #273951;   /* Form labels, secondary headings */
```

#### Accents (decorative only — NOT for buttons/links)
```css
--m-ruby:             #ea2261;   /* Decorative accent, gradients */
--m-magenta:          #f96bee;   /* Gradient middle, decorative */
--m-magenta-light:    #ffd7ef;   /* Tinted surface for badges */
```

#### Semantic
```css
--m-success:          #15be53;
--m-success-text:     #108c3d;
--m-warning:          #9b6829;
--m-error:            #ea2261;
```

#### Borders & Surfaces
```css
--m-border:           #e5edf5;   /* Standard border for cards, dividers */
--m-border-purple:    #b9b9f9;   /* Active/selected state borders */
--m-border-soft:      #d6d9fc;   /* Subtle purple-tinted borders */
```

#### Shadows (blue-tinted — the Stripe signature)
```css
--m-shadow-standard:  rgba(50,50,93,0.25) 0px 30px 45px -30px, rgba(0,0,0,0.1) 0px 18px 36px -18px;
--m-shadow-ambient:   rgba(23,23,23,0.08) 0px 15px 35px 0px;
--m-shadow-subtle:    rgba(23,23,23,0.06) 0px 3px 6px;
--m-shadow-deep:      rgba(3,3,39,0.25) 0px 14px 21px -14px, rgba(0,0,0,0.1) 0px 8px 17px -8px;
```

### Typography Scale

| Role | Size | Weight | Line Height | Letter Spacing | Color |
|------|------|--------|-------------|----------------|-------|
| Display Hero | 56px | 300 | 1.03 | -1.4px | #061b31 |
| Display Large | 48px | 300 | 1.15 | -0.96px | #061b31 |
| Section Heading | 32px | 300 | 1.10 | -0.64px | #061b31 |
| Sub-heading Large | 26px | 300 | 1.12 | -0.26px | #061b31 |
| Sub-heading | 22px | 300 | 1.10 | -0.22px | #061b31 |
| Body Large | 18px | 300 | 1.40 | normal | #64748d |
| Body | 16px | 300-400 | 1.40 | normal | #64748d |
| Button | 16px | 400 | 1.00 | normal | (varies) |
| Button Small | 14px | 400 | 1.00 | normal | (varies) |
| Link/Nav | 14px | 400 | 1.00 | normal | #061b31 |
| Caption | 13px | 400 | normal | normal | #64748d |
| Caption Small | 12px | 300-400 | 1.33 | normal | #64748d |
| Code | 12px Source Code Pro | 500 | 2.00 | normal | (varies) |

CRITICAL: Weight 300 for all headlines and body. This is the signature — light, confident, anti-convention. Weight 400 only for buttons, links, navigation. NEVER use 600-700 for Geist headlines.

### Components

#### Buttons

**Primary (Purple):**
```css
background: #533afd;
color: #ffffff;
padding: 8px 16px;
border-radius: 4px;
font: Geist 16px weight 400;
font-feature-settings: "ss01";
/* Hover: */ background: #4434d4;
```

**Ghost / Outlined:**
```css
background: transparent;
color: #533afd;
padding: 8px 16px;
border-radius: 4px;
border: 1px solid #b9b9f9;
/* Hover: */ background: rgba(83,58,253,0.05);
```

**Neutral Ghost:**
```css
background: transparent;
color: rgba(16,16,16,0.3);
padding: 8px 16px;
border-radius: 4px;
outline: 1px solid #d4dee9;
```

#### Cards
```css
background: #ffffff;
border: 1px solid #e5edf5;
border-radius: 6px;
box-shadow: rgba(50,50,93,0.25) 0px 30px 45px -30px, rgba(0,0,0,0.1) 0px 18px 36px -18px;
/* Hover: shadow intensifies */
```

#### Inputs
```css
border: 1px solid #e5edf5;
border-radius: 4px;
color: #061b31;
/* Focus: */ border-color: #533afd;
/* Label: */ color: #273951; font-size: 14px;
/* Placeholder: */ color: #64748d;
```

#### Navigation (sticky)
```css
background: #ffffff;
backdrop-filter: blur(12px);
/* Links: */ font: Geist 14px weight 400; color: #061b31;
/* CTA: */ #533afd background, white text, 4px radius;
/* Nav container: */ border-radius: 6px;
```

#### Dark Sections (brand immersion)
```css
background: #1c1e54;
/* Heading: */ color: #ffffff;
/* Body: */ color: rgba(255,255,255,0.7);
/* Cards inside: */ border: rgba(255,255,255,0.1); border-radius: 6px;
```

#### Success Badge
```css
background: rgba(21,190,83,0.2);
color: #108c3d;
padding: 1px 6px;
border-radius: 4px;
border: 1px solid rgba(21,190,83,0.4);
font: Geist 10px weight 300;
```

### Layout Rules
- Max content width: ~1080px centered
- Border radius: 4px (buttons, inputs, badges), 5px (standard cards), 6px (comfortable), 8px (featured)
- NEVER use pill shapes or radius > 8px on marketing pages
- Section rhythm: white sections alternate with dark brand (#1c1e54) sections
- Shadows are blue-tinted: always use rgba(50,50,93,...) as the primary shadow color
- Use #061b31 (deep navy) for headings, NEVER #000000

### Marketing Page Do's and Don'ts

DO:
- Use weight 300 for all headlines — lightness is luxury
- Apply blue-tinted shadows for all elevated elements
- Use #533afd purple as the primary interactive/CTA color
- Use font-feature-settings "ss01" on all Geist text
- Keep border-radius between 4px-8px
- Alternate white and dark (#1c1e54) sections for rhythm

DON'T:
- Don't use weight 600-700 for Geist headlines
- Don't use large border-radius (12px+, pill shapes)
- Don't use neutral gray shadows — always blue-tinted
- Don't use pure black (#000000) for headings — use #061b31
- Don't use orange, yellow, or warm accents for interactive elements — purple is primary
- Don't use ruby/magenta accents for buttons — they are decorative/gradient ONLY

---

## DESIGN SYSTEM B: APP PAGES (Cursor-Inspired)

**Applies to**: `/app/dashboard`, `/app/generate`, `/app/explore`, `/app/history`, `/app/settings/*` — everything behind login

### Font Families (open-source equivalents)
```
Display/Headlines:   "Satoshi", system-ui, sans-serif
Body/Editorial:      "Instrument Serif", ui-serif, Georgia, serif
Code/Technical:      "JetBrains Mono", ui-monospace, monospace
UI/System:           system-ui, -apple-system, sans-serif
```
Load Satoshi from fontshare.com. Instrument Serif and JetBrains Mono from Google Fonts.

### Color Palette

#### Primary
```css
--a-dark:             #26251e;   /* Primary text — warm near-black with yellow undertone */
--a-cream:            #f2f1ed;   /* Page background — warm off-white */
--a-light:            #e6e5e0;   /* Secondary surface, card fills */
--a-white:            #ffffff;   /* Sparingly, for max contrast elements */
--a-black:            #000000;   /* Minimal use, code/console contexts only */
```

#### Accent
```css
--a-orange:           #f54e00;   /* Brand accent — CTAs, active links, brand moments */
--a-gold:             #c08532;   /* Secondary accent, premium/highlighted contexts */
```

#### Semantic
```css
--a-error:            #cf2d56;   /* Warm crimson-rose (also button hover text color!) */
--a-success:          #1f8a65;   /* Muted teal-green */
```

#### AI Feature Colors (for generation phases, timelines)
```css
--a-thinking:         #dfa88f;   /* Warm peach — "processing" state */
--a-search:           #9fc9a2;   /* Soft sage — search/grep operations */
--a-read:             #9fbbe0;   /* Soft blue — reading/loading operations */
--a-edit:             #c0a8dd;   /* Soft lavender — editing/writing operations */
```

#### Surface Scale (warm cream spectrum)
```css
--a-surface-100:      #f7f7f4;   /* Lightest — barely tinted */
--a-surface-200:      #f2f1ed;   /* Primary page background */
--a-surface-300:      #ebeae5;   /* Button default background */
--a-surface-400:      #e6e5e0;   /* Card backgrounds */
--a-surface-500:      #e1e0db;   /* Tertiary, deeper emphasis */
```

#### Borders (warm brown via oklab, with rgba fallback)
```css
--a-border:           rgba(38, 37, 30, 0.1);    /* Standard border, 10% warm brown */
--a-border-medium:    rgba(38, 37, 30, 0.2);    /* Emphasized border */
--a-border-strong:    rgba(38, 37, 30, 0.55);   /* Strong borders, table rules */
--a-border-solid:     #26251e;                   /* Full-opacity dark border */
```

#### Shadows
```css
--a-shadow-elevated:  rgba(0,0,0,0.14) 0px 28px 70px, rgba(0,0,0,0.1) 0px 14px 32px;
--a-shadow-ambient:   rgba(0,0,0,0.02) 0px 0px 16px, rgba(0,0,0,0.008) 0px 0px 8px;
--a-shadow-focus:     rgba(0,0,0,0.1) 0px 4px 12px;
```

### Typography Scale

| Role | Font | Size | Weight | Line Height | Letter Spacing |
|------|------|------|--------|-------------|----------------|
| Display Hero | Satoshi | 72px | 400 | 1.10 | -2.16px |
| Section Heading | Satoshi | 36px | 400 | 1.20 | -0.72px |
| Sub-heading | Satoshi | 26px | 400 | 1.25 | -0.325px |
| Title Small | Satoshi | 22px | 400 | 1.30 | -0.11px |
| Body Serif | Instrument Serif | 19.2px | 500 | 1.50 | normal |
| Body Serif SM | Instrument Serif | 17.28px | 400 | 1.35 | normal |
| Body Sans | Satoshi | 16px | 400 | 1.50 | normal |
| Button Label | Satoshi | 14px | 400 | 1.00 | normal |
| Caption | Satoshi | 11px | 400-500 | 1.50 | normal |
| System Heading | system-ui | 20px | 700 | 1.55 | normal |
| System Caption | system-ui | 13px | 500-600 | 1.33 | normal |
| System Micro | system-ui | 11px | 500 | 1.27 | 0.048px UPPERCASE |
| Code Body | JetBrains Mono | 12px | 400 | 1.67 | normal |
| Code Small | JetBrains Mono | 11px | 400 | 1.33 | -0.275px |

CRITICAL: Satoshi uses negative letter-spacing at display sizes: -2.16px at 72px, -0.72px at 36px, -0.325px at 26px, normal at 16px and below. Weight 400 almost exclusively — hierarchy comes from size and tracking, not weight.

### Components

#### Buttons

**Primary (Warm Surface):**
```css
background: #ebeae5;
color: #26251e;
padding: 10px 12px 10px 14px;
border-radius: 8px;
font: Satoshi 14px weight 400;
/* Hover: */ color: #cf2d56;  /* TEXT shifts to warm crimson — signature interaction */
/* Focus: */ box-shadow: rgba(0,0,0,0.1) 0px 4px 12px;
```

**Secondary Pill:**
```css
background: #e6e5e0;
color: rgba(38, 37, 30, 0.6);
padding: 3px 8px;
border-radius: 9999px;  /* Full pill — ONLY in app, never marketing */
/* Hover: */ color: #cf2d56;
```

**Ghost (Transparent):**
```css
background: rgba(38, 37, 30, 0.06);
color: rgba(38, 37, 30, 0.55);
padding: 6px 12px;
border-radius: 8px;
```

**Accent (AI features, primary CTAs):**
```css
background: #f54e00;
color: #ffffff;
padding: 10px 14px;
border-radius: 8px;
/* Hover: */ opacity: 0.9;
```

#### Cards & Containers
```css
background: #e6e5e0;  /* or #f2f1ed for lighter variant */
border: 1px solid rgba(38, 37, 30, 0.1);
border-radius: 8px;
/* Elevated: */ box-shadow: rgba(0,0,0,0.14) 0px 28px 70px, rgba(0,0,0,0.1) 0px 14px 32px;
/* Hover: */ shadow intensifies;
```

#### Text Inputs
```css
background: transparent;
color: #26251e;
padding: 8px;
border: 1px solid rgba(38, 37, 30, 0.1);
border-radius: 8px;
/* Focus: */ border-color: rgba(38, 37, 30, 0.2);
/* Placeholder: */ color: rgba(38, 37, 30, 0.4);
```

#### Chat Message Bubbles
```css
/* User message: */
background: #26251e;
color: #ffffff;
border-radius: 8px 8px 2px 8px;
font: Satoshi 16px/1.50 weight 400;
/* Aligned right */

/* Assistant message: */
background: #ffffff;
color: #26251e;
border: 1px solid rgba(38, 37, 30, 0.1);
border-radius: 8px 8px 8px 2px;
font: Satoshi 16px/1.50 weight 400;
/* Aligned left */

/* System/generation progress messages: */
background: #ebeae5;
border: 1px solid rgba(38, 37, 30, 0.1);
border-radius: 8px;
/* Full width, centered */
```

#### Generation Phase Progress (in chat)
```css
/* Container: */
background: #ebeae5;
border: 1px solid rgba(38, 37, 30, 0.1);
border-radius: 8px;
padding: 16px;

/* Phase labels: */
font: system-ui 11px weight 500 UPPERCASE;
letter-spacing: 0.048px;

/* Phase states: */
/* Done (✓): */ color: #1f8a65;
/* Active (●): */ color: #f54e00; /* subtle pulse animation */
/* Pending (○): */ color: rgba(38, 37, 30, 0.3);

/* Progress bar: */
background: rgba(38, 37, 30, 0.1);  /* track */
fill: #f54e00;  /* progress */
border-radius: 4px;
height: 4px;
```

#### Quality Score Badge
```css
/* Score number: */
font: Satoshi 32px weight 400;
/* Color based on score: */
/* >80: */ color: #1f8a65;
/* 60-80: */ color: #c08532;
/* <60: */ color: #cf2d56;

/* Label below: */
font: JetBrains Mono 11px weight 500 UPPERCASE;
letter-spacing: 0.5px;
color: rgba(38, 37, 30, 0.55);
```

#### App Navigation Sidebar
```css
/* Sidebar: */
width: 260px;
background: #ffffff;
border-right: 1px solid rgba(38, 37, 30, 0.1);

/* Nav items: */
font: Satoshi 14px weight 400;
color: rgba(38, 37, 30, 0.55);     /* default */
color: #26251e;                     /* active */
background: #f2f1ed;               /* active item background */
border-left: 2px solid #f54e00;    /* active indicator */
padding: 8px 12px;
border-radius: 0 6px 6px 0;

/* Icons: Lucide React, 18px, stroke-width 1.5 */
/* Hover: */ background: #f7f7f4;
transition: all 150ms ease;

/* Logo area: */
padding: 20px 16px;
/* SynthFlow logo in charcoal (#3d4043) */

/* Plan indicator at bottom: */
font: system-ui 12px weight 500;
color: rgba(38, 37, 30, 0.55);
/* [Upgrade] button: accent orange */
```

#### Data Table (for history, data preview)
```css
/* Table container: */
background: #ffffff;
border: 1px solid rgba(38, 37, 30, 0.1);
border-radius: 8px;
overflow: hidden;

/* Header row: */
background: #f7f7f4;
font: system-ui 11px weight 600 UPPERCASE;
letter-spacing: 0.048px;
color: rgba(38, 37, 30, 0.55);
border-bottom: 1px solid rgba(38, 37, 30, 0.1);

/* Body cells: */
font: Satoshi 14px weight 400;
color: #26251e;
padding: 10px 14px;
border-bottom: 1px solid rgba(38, 37, 30, 0.06);

/* Numeric cells: */
font: JetBrains Mono 13px weight 400;
font-feature-settings: "tnum";
```

### Layout Rules

#### Spacing Scale
Base 8px. Fine sub-8px for micro-alignment: 1.5, 2, 2.5, 3, 4, 5, 6px.
Standard: 8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64, 80, 96px.

#### Border Radius Scale
- 2px: inline elements, code spans
- 4px: compact buttons, progress bars, micro elements
- 6px: nav items (right side), tags
- 8px: primary buttons, cards, inputs, chat bubbles — THE WORKHORSE
- 9999px: pill tags/filters (app pages ONLY, never marketing)

#### Depth & Elevation
| Level | Treatment | Use |
|-------|-----------|-----|
| Flat | No shadow | Page background, text blocks |
| Border | 1px solid rgba(38,37,30,0.1) | Standard cards, containers |
| Ambient | rgba(0,0,0,0.02) 0px 0px 16px | Floating elements, subtle glow |
| Elevated | rgba(0,0,0,0.14) 0px 28px 70px, rgba(0,0,0,0.1) 0px 14px 32px | Modals, popovers, dropdowns |
| Focus | rgba(0,0,0,0.1) 0px 4px 12px | Button focus state |

NO harsh shadows. Large blur values (28px, 70px) create diffused atmospheric lift.

#### App Shell Layout
```
Sidebar: 260px wide, bg #ffffff, border-right rgba(38,37,30,0.1)
Main content: bg #f2f1ed, fills remaining width
Content max-width: ~1200px centered within main area
Sidebar collapses to icon-only on mobile (<768px)
```

### Responsive Breakpoints
```
sm:  600px
md:  768px
lg:  900px
xl:  1280px
```

### App Pages Do's and Don'ts

DO:
- Use warm tones everywhere — #f2f1ed background, #26251e text, never pure white/black for primary surfaces
- Negative letter-spacing on Satoshi display sizes (-2.16px at 72px scaling down)
- Use #cf2d56 text color on button hover — the warm crimson shift is signature
- Use Instrument Serif for longer body text and editorial moments
- Use JetBrains Mono for all code, data values, and UPPERCASE micro labels
- Use the AI feature colors (peach, sage, blue, lavender) for generation phases
- Use large blur shadows (28px, 70px) for elevation
- Pill shapes (9999px) are OK for tags/filters in app

DON'T:
- Don't use pure white (#ffffff) as the main app background — use #f2f1ed
- Don't use cool gray borders — always warm brown rgba(38,37,30,...)
- Don't use heavy font weights (600-700) on Satoshi — hierarchy from size, not weight
- Don't use purple (#533afd) in the app — that belongs to marketing pages
- Don't use blue-tinted shadows in the app — those belong to marketing pages
- Don't mix Stripe components into app pages or Cursor components into marketing pages

---

## Shared Brand Elements (Both Design Systems)

| Element | Marketing (Stripe) | App (Cursor) |
|---------|-------------------|--------------|
| Logo color | #3d4043 charcoal | #3d4043 charcoal |
| Primary CTA color | #533afd purple | #f54e00 orange |
| Background | #ffffff white | #f2f1ed warm cream |
| Text color | #061b31 deep navy | #26251e warm brown |
| Heading font | Geist weight 300 | Satoshi weight 400 |
| Body font | Geist weight 300 | Instrument Serif / Satoshi |
| Code font | Source Code Pro | JetBrains Mono |
| Card radius | 4-6px | 8px |
| Shadow tone | Blue-tinted rgba(50,50,93,...) | Neutral rgba(0,0,0,...) |
| Button hover | Background color change | Text color shifts to #cf2d56 |
| Pill shapes | NEVER | OK for tags/filters |

---

## Running Tests

```bash
# Engine tests
cd engine && pytest tests/ -v --cov=synthflow --cov-report=term-missing

# Backend tests
cd backend && pytest tests/ -v

# Frontend tests
cd frontend && npm test

# Type checking
cd engine && mypy synthflow/ --strict
cd frontend && npx tsc --noEmit
```

## Build Order

Follow the Development Plan (docs/DEVELOPMENT_PLAN.md) exactly.
Build one component → write tests → run tests → fix → commit → proceed.

Current progress: [UPDATE THIS AS YOU BUILD]
- [ ] Layer 0: Scaffold
- [ ] Layer 1: Engine Foundation
- [ ] Layer 2: Engine Subsystems
- [ ] Layer 3: Backend Auth + DB
- [ ] Layer 4: Backend Core API
- [ ] Layer 5: Backend Business
- [ ] Layer 6: Frontend Foundation
- [ ] Layer 7: Frontend Core
- [ ] Layer 8: Frontend Settings
- [ ] Layer 9: Integration

## Key Data Contracts

### IntentObject (engine input)
```python
{
    "domain": "healthcare",
    "sub_domain": "cardiology",
    "region": {"country": "India", "state_province": "Maharashtra"},
    "row_count": 5000,
    "seed": 42,
    "implied_columns": ["patient_id", "name", "age", ...]
}
```

### GenerationResult (engine output)
```python
{
    "session_id": "uuid",
    "dataframe": pd.DataFrame,
    "schema": SchemaDefinition,
    "validation_report": {...},
    "quality_report": {...},
    "privacy_report": {...},
    "generated_code": "string",
    "intent": IntentObject
}
```

### WebSocket Messages (frontend <-> backend)
```json
// Client -> Server
{"type": "message", "content": "Generate 5000 records", "conversation_id": "uuid"}

// Server -> Client (streaming text)
{"type": "text_chunk", "content": "I'll generate..."}

// Server -> Client (generation progress)
{"type": "phase_update", "phase": 3, "progress": 0.33, "message": "Designing schema..."}

// Server -> Client (generation complete)
{"type": "generation_done", "generation_id": "uuid", "quality_score": 94.2, "preview_rows": [...]}
```

## Environment Variables

### Backend (.env)
```
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/synthflow
REDIS_URL=redis://default:pass@host:6379
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
JWT_SECRET=random-256-bit-secret
ENCRYPTION_KEY=random-256-bit-key
RAZORPAY_KEY_ID=rzp_...
RAZORPAY_KEY_SECRET=...
STRIPE_SECRET_KEY=sk_...
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```
