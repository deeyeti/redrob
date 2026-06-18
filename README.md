# Team Antigravity — Redrob Hackathon Submission

> **Challenge**: Redrob India Runs Data & AI Challenge  
> **Task**: Intelligent Candidate Discovery & Ranking — Senior AI Engineer role  
> **Team**: Antigravity

---

## 🎯 Problem Statement

Given a dataset of 100,000 candidate profiles (`candidates.jsonl`) and a job description for a **Senior AI Engineer** at a Series A AI-native company, rank the **top 100 best-fit candidates** and output them with scores and reasoning.

**Evaluation**: NDCG@10 (50%) · NDCG@50 (30%) · MAP (15%) · P@10 (5%)

---

## 🏗️ Approach

A **multi-signal JD-aware ranker** built entirely in Python with no external API calls, no GPU, and no pre-trained models — processes 100K candidates in **~10 seconds** on a modern multi-core CPU using parallel processing.

### Scoring Architecture

| Component | Weight | What it measures |
|---|---|---|
| **Skill Match** | 35% | Hand-curated JD taxonomy + Pure-Python TF-IDF matching against the full JD text + Skill synonym normalization. Weighted by proficiency × duration |
| **Title/Role** | 17% | 4-tier taxonomy: ML/AI/NLP Engineer (1.0) → SWE/Backend (0.55) → Frontend/QA (0.15) → Non-tech (0.0) |
| **YoE Fit** | 13% | Sweet spot 6–8 yrs (JD says 5–9); tapers off beyond that |
| **Behavioral Signals** | 14% | `recruiter_response_rate` (35%), `interview_completion_rate` (25%), GitHub score (15%), recency (10%), saved by recruiters (10%), profile completeness (5%) |
| **Career Depth** | 7% | ML keyword breadth + Regex-based quantified impact detection (e.g. 10x, 20ms, 5 million) + production deployment signals ("shipped", "A/B test") + description richness |
| **Company Context** | 6% | Product/tech company ratio vs consulting; tech industry (FinTech, EdTech, SaaS, AI) rewarded; company size preference (scaleup sweet spot) |
| **Education** | 4% | Institution tier (tier_1=IIT/IISc/top global) + field relevance (CS/AI/Math) |
| **Availability** | 4% | Notice period + open-to-work flag + location (Pune/Noida preferred) + work mode preference |

### Honeypot Detection

Candidates are **hard-excluded** if:
- Career dates are reversed (`end_date < start_date`)
- ≥8 skills claim "expert" proficiency with `duration_months = 0`
- `years_of_experience > 50`

Candidates are **soft-penalised** (score × 0.1) if:
- Entire career history is at known consulting giants (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, etc.)

### Reasoning Generation

Every candidate gets a **specific, factual reasoning string** that mentions:
- Actual title + YoE
- Top 3 matched JD skills (with proficiency evidence)
- Company/industry context (product co vs consulting)
- Production deployment evidence found in career text
- Key behavioral signals (response rate, GitHub score)
- Location + availability signals
- Honest concerns where relevant (consulting-heavy career, YoE outside band, low interview completion)

---

## 🚀 How to Run

### Requirements

```bash
pip install -r requirements.txt
# Only pandas is needed — everything else is stdlib
```

### Generate submission

```bash
python rank.py --candidates candidates.jsonl --out team_antigravity.csv
```

**Expected output**: ~10-15 seconds on a modern multi-core CPU, `team_antigravity.csv` with 100 ranked candidates.

### Validate submission format

```bash
python validate_submission.py team_antigravity.csv
# → Submission is valid.
```

### Run in notebook

Open `antigravity_pipeline.ipynb` — it calls `rank.py` directly and includes format validation and an inspection cell.

---

## 📁 File Structure

```
.
├── rank.py                          # Main ranking pipeline (standalone)
├── antigravity_pipeline.ipynb       # Jupyter notebook wrapper
├── validate_submission.py           # Official format checker (provided by Redrob)
├── candidate_schema.json            # Official candidate schema (provided by Redrob)
├── sample_candidates.json           # 50-candidate sample for testing
├── sample_submission.csv            # Example submission format
├── submission_metadata.yaml         # Team metadata for portal submission
├── requirements.txt                 # Python dependencies
└── team_antigravity.csv             # Our final submission
```

> `candidates.jsonl` (487 MB) is excluded from the repo via `.gitignore`. Place it in the root before running.

---

## ⚙️ Compute Constraints

| Constraint | Requirement | Our Result |
|---|---|---|
| Runtime | ≤ 5 min | ~10-15 seconds (parallelized) |
| Memory | ≤ 16 GB RAM | < 2 GB |
| Compute | CPU only | ✅ CPU only |
| Network | No external calls | ✅ No API calls |

---

## 📊 Submission Stats

- **Total candidates processed**: 100,000
- **Honeypots filtered**: 0
- **Soft-disqualified (consulting-only)**: ~3,700
- **Score range**: 0.650 – 0.759
- **Top candidate**: Lead AI Engineer, 6.7 yrs, FinTech product company, strong production retrieval system evidence

---

## 🔬 Design Decisions

**Why not use LLMs for ranking?**  
The spec explicitly forbids external API calls during ranking and requires ≤5 min CPU-only runtime. Calling an LLM per candidate would be orders of magnitude too slow.

**Why not TF-IDF / BM25 against the JD?**  
We tried a hybrid approach but found that structured feature extraction (skill taxonomy + behavioral signals) outperforms pure text similarity on this dataset because: (a) the skill names are already normalized in the schema, (b) behavioral signals like `recruiter_response_rate` are highly predictive and have no text representation, and (c) career descriptions are synthetic and sometimes mismatched with the skill list.

**Why scan career descriptions separately from the skills list?**  
Candidates often mention tools in their descriptions (e.g., "built a FAISS-based retrieval pipeline") without listing them as formal skills. The `career_depth` component catches this evidence and rewards it — separately from the structured skill match.

**On the consulting disqualifier:**  
The JD explicitly states candidates from pure consulting firms (TCS, Infosys, Wipro, etc.) are a poor fit. We apply a graded penalty (0.1× multiplier for the hard case, score reduction via `company_context` for mixed careers) rather than a binary exclude, to allow edge cases through.
