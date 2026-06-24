# Team Eclectic — Redrob Hackathon Submission

> **Challenge**: Redrob India Runs Data & AI Challenge  
> **Task**: Intelligent Candidate Discovery & Ranking — Senior AI Engineer role  
> **Team**: Eclectic

---

## 🎯 Problem Statement

Given a dataset of 100,000 candidate profiles (`candidates.jsonl`) and a job description for a **Senior AI Engineer** at a Series A AI-native company, rank the **top 100 best-fit candidates** and output them with scores and explicit reasoning.

**Evaluation**: NDCG@10 (50%) · NDCG@50 (30%) · MAP (15%) · P@10 (5%)

---

## 🏗️ Approach

A **multi-signal JD-aware ranker** built with intelligent evaluation layers, combining deterministic heuristics, TF-IDF, and multiple NLP/ML-based semantic signals. Processes all 100K candidates in **under 10 seconds** on a standard modern CPU.

---

## ⚙️ Pipeline Architecture

Every candidate is evaluated by a **weighted 8-component scoring engine** combined with a **regex multiplier** applied simultaneously.

#### 🧠 Core Scoring Architecture (8 Components)

| Component | Weight | Description |
|---|---|---|
| **Skill Match** | 35% | Keyword + TF-IDF scoring against 80+ JD skills |
| **Title / Role Fit** | 17% | 4-tier role taxonomy aligned to JD requirements |
| **YoE Fit** | 13% | Optimal band of 6–8 years, steep penalties outside |
| **Behavioral Signals** | 14% | 6 platform signals (response, GitHub, recency, saves) |
| **Career Depth** | 7% | ML keyword breadth + production deployment evidence |
| **Company Context** | 6% | Product vs. consulting background, company size |
| **Education Tier** | 4% | Tier 1–4 institution + relevant field bonus |
| **Availability** | 4% | Notice period, location fit, open-to-work flag |

**Final Score** = `raw_score × regex_multiplier`

---

#### 📐 Skill Match — Component Detail

The skill score is built from three sub-signals combined as `0.60 × core_norm + 0.15 × nice_norm + 0.25 × tfidf_norm`:

1. **Core Skills (60%)** — 80+ JD-specific skill keywords (embeddings, FAISS, RAG, LoRA, BM25, etc.), each with a base weight (0.3–1.0). Proficiency level, duration in months, and assessment scores all modulate each keyword's weight.
2. **Nice-to-Have Skills (15%)** — 22 adjacent skills (Docker, Kafka, XGBoost, etc.) at reduced weight.
3. **TF-IDF Similarity (25%)** — Token overlap between the candidate's full text blob (skills + titles + descriptions) and the JD text, using term frequency scaled by JD frequency.

Skill synonyms are normalised before matching (`llm → large language model`, `k8s → kubernetes`, etc.).

---

#### 🏷️ Title / Role Taxonomy

Titles are classified into 4 tiers:

| Tier | Score | Examples |
|---|---|---|
| **Tier 1** | 1.00 | ML Engineer, AI Engineer, NLP Engineer, Ranking Engineer, Data Scientist |
| **Tier 2** | 0.55 | Software Engineer, Backend Engineer, Data Engineer, Cloud Engineer |
| **Tier 3** | 0.15 | Frontend Engineer, Mobile Developer, Java Developer, QA Engineer |
| **Tier 0** | 0.00 | HR Manager, Sales Executive, Civil Engineer, Graphic Designer |

The effective tier is the **best** of the current title and entire career history.

---

#### 📊 Behavioral Signals — 6 Platform Metrics

| Signal | Weight |
|---|---|
| `recruiter_response_rate` | 35% |
| `interview_completion_rate` | 25% |
| `github_activity_score` | 15% |
| `last_active_date` recency | 10% |
| `saved_by_recruiters_30d` | 10% |
| `profile_completeness_score` | 5% |

---

#### 🏭 Career Depth — 2 Dimensions

1. **ML Keyword Breadth**: Unique domain terms found in career descriptions (embeddings, FAISS, transformer, RAG, etc.). 10+ unique terms = full score.
2. **Production Evidence**: Count of impact phrases (deployed, shipped, at scale, 50ms latency, A/B test, NDCG improved, etc.) + quantified metric claims (%, x multipliers, $ figures). 6+ signals = full score.

---

#### 🏢 Company Context — Product vs. Consulting

- **Rewards**: Product/tech company background (SaaS, AI-native, fintech), scaleup size (51–200 employees), current industry being tech-adjacent.
- **Penalises**: Career spent at known consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) up to a 70% penalty.

---

#### 🌐 Availability Scoring

- Notice period: ≤30d → 1.0, ≤60d → 0.6, ≤90d → 0.35, >90d → 0.1
- Location: Pune/Noida → 1.0, Bangalore/Mumbai/Hyderabad → 0.75, willing to relocate → 0.6
- Bonuses: Open-to-work flag (+0.3), willing to relocate (+0.1), hybrid/flexible mode preference (+0.1)

---

#### 🧬 Regex Enrichment Layer

The base score is multiplied by a context-aware regex signal (`score_nlp_context`) which applies caveat detection and positive boosting:

- **Penalty (→ 0.3x)**: If the candidate has weak ML skills AND phrases like "experimenting with ChatGPT", "transitioning to AI", "no formal ML experience", "hobbyist side project", or "taking courses in deep learning" appear in their full text.
- **Bonus (+4–8%)**: If 1–3+ patterns matching strong ownership (`built/deployed/architected + retrieval/embedding/RAG/LLM`), quantified impact (`improved NDCG by 20%`), or system-design specifics (`hybrid BM25-dense retrieval`, `fine-tuned with LoRA`) are found.

**Hard Disqualifiers**:
- Entire career at consulting firms (2+ jobs) → multiplier 0.10
- Tier-0 non-tech title (graphic designer, HR manager) with no ML signal in full text → multiplier 0.05

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Submission

```bash
# Full run (100K candidates)
python rank.py --candidates candidates.jsonl --out team_eclectic.csv

# Quick test run (50-candidate sample)
python rank.py --candidates sample_candidates.json --out team_eclectic.csv
```

**Expected runtime**: under 10 seconds on a standard modern CPU (Stage 1: ~50s, Stage 2 reranking of Top 1K: ~40s).

### Validate Output

```bash
python validate_submission.py team_eclectic.csv
```

### Debug Mode

```bash
python rank.py --candidates candidates.jsonl --out team_eclectic.csv --debug
```

Debug mode outputs all 8 component scores + the NLP multiplier as extra columns.

---

## 📁 File Structure

```
.
├── rank.py                       # Main ranking pipeline (1,314 lines)
├── eclectic_pipeline.ipynb       # Jupyter notebook wrapper (mirrors rank.py)
├── team_eclectic.csv             # Final output submission (top 100 candidates)
├── validate_submission.py        # Official format validator
├── sample_candidates.json        # 50-candidate test dataset
├── candidates.jsonl              # Full 100K candidate dataset
├── sample_submission.csv         # Reference submission format
├── candidate_schema.json         # Candidate profile JSON schema
├── requirements.txt              # Python dependencies
├── submission_metadata.yaml      # Team + submission metadata
├── job_description.docx          # Original JD (reference)
├── redrob_signals_doc.docx       # Signal definitions (reference)
└── submission_spec.docx          # Output format specification
```

---

## 🔧 Design Decisions

### Why Not Use a Generative LLM?
The hackathon enforces a **≤5 minute CPU-only constraint** on a standard machine. Generative LLMs (GPT-4, LLaMA, Mistral) require GPU and produce 1 token/second on CPU, making them impossible to use on 100K candidates. Our architecture relies solely on highly optimized Python logic, dictionary lookups, and regex patterns, ensuring instant processing.

### Why Rules, Not a Trained Model?

There is no labelled ground truth data for this specific JD. A rules engine derived directly from the JD text (skill taxonomy, title tiers, YoE bands) provides high-precision signals without needing training data.


