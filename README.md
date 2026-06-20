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

A **multi-signal, two-stage JD-aware ranker** built with intelligent evaluation layers, combining deterministic heuristics, TF-IDF, and multiple NLP/ML-based semantic signals. Processes all 100K candidates in **~90 seconds** on a standard modern CPU.

---

## ⚙️ Two-Stage Pipeline Architecture

### Stage 1 — Fast Scan (All 100K Candidates)

Every candidate is evaluated by a **weighted 8-component scoring engine** combined with a **3-layer NLP multiplier** applied simultaneously.

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

**Final Stage 1 Score** = `raw_score × nlp_multiplier × minilm_sim_multiplier`

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

#### 🧬 NLP Enrichment Layers (Stage 1 Multiplier)

The base score is multiplied by a context-aware NLP signal (`score_nlp_context`) which applies three layers:

**Layer 1 — Regex Caveat Detection & Positive Boosting**
- **Penalty (→ 0.3x)**: If the candidate has weak ML skills AND phrases like "experimenting with ChatGPT", "transitioning to AI", "no formal ML experience", "hobbyist side project", or "taking courses in deep learning" appear in their full text.
- **Bonus (+4–8%)**: If 1–3+ patterns matching strong ownership (`built/deployed/architected + retrieval/embedding/RAG/LLM`), quantified impact (`improved NDCG by 20%`), or system-design specifics (`hybrid BM25-dense retrieval`, `fine-tuned with LoRA`) are found.

**Layer 2 — spaCy Verb-Object Dependency Parsing**
- The Cython-optimized spaCy dependency parser processes up to **3,000 characters** of the combined profile and career descriptions (extended from 800 chars to capture deeper resume content).
- Extracts verb → object pairs. Strong **ownership verbs** (`built`, `architected`, `shipped`, `deployed`, `trained`, `fine-tuned`) pointing at **JD nouns** (`pipeline`, `system`, `embedding`, `ranker`, `vector`, `RAG`, `LLM`) add a +2–5% multiplier.
- Weak **familiarity verbs** (`explored`, `experimented`, `learned`, `studied`, `read`) pointing at the same nouns with no ownership counterpart subtract 5%.

**Layer 3 — MiniLM Bi-Encoder Semantic Similarity**
- All candidate summaries + career descriptions (up to 512 chars) are batch-encoded with `all-MiniLM-L6-v2`.
- Cosine similarity to a hand-crafted JD-derived ideal embedding is computed via a dot product (vecs are pre-normalised).
- Similarity is mapped to a tight multiplier: `sim=0.45 → 1.00x` (neutral), `sim=0.75+ → 1.07x` (capped), `sim<0.33 → 0.93x` (mild penalty). This fine-tunes rankings without overriding base score differentiation.

**Hard Disqualifiers**:
- Entire career at consulting firms (2+ jobs) → multiplier 0.10
- Tier-0 non-tech title (graphic designer, HR manager) with no ML signal in full text → multiplier 0.05

---

### Stage 2 — Deep Re-Ranking (Top 1,000 Candidates)

After Stage 1 filters and ranks all 100K candidates, the **Top 1,000** are passed to two computationally heavier NLP models for precise semantic re-scoring.

#### 🔁 Cross-Encoder Re-Ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

Unlike the bi-encoder (which encodes JD and candidate separately), the Cross-Encoder evaluates the JD ideal text **paired with** each candidate's 1,500-character profile at the same time, enabling deep cross-attention between the two.

- Outputs a relevance logit per candidate.
- Logit is passed through a sigmoid function and remapped to `[0.8x, 1.2x]` multiplicative range.
- Run in batches of 32 for CPU efficiency.

#### 🔍 Implicit Skill Extraction (`cross-encoder/nli-distilroberta-base` — Zero-Shot NLI)

Candidates frequently describe using skills in their narrative without ever listing them formally. The Zero-Shot classifier evaluates each candidate's career text for **entailment** of 4 critical JD concepts:

- `"Vector Databases"`
- `"LLM Fine-tuning"`
- `"Learning to Rank"`
- `"Semantic Search"`

If confidence > 60% that the text entails a concept, it is treated as an implicit skill detected. Each implicit skill found adds **+5% to the final score** (up to +20% for all 4).

The reasoning string in the output CSV is also updated to append `implicitly detected skills: Vector Databases, LLM Fine-tuning` so recruiters can see exactly why the candidate was boosted.

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Generate Submission

```bash
# Full run (100K candidates)
python rank.py --candidates candidates.jsonl --out team_eclectic.csv

# Quick test run (50-candidate sample)
python rank.py --candidates sample_candidates.json --out team_eclectic.csv
```

**Expected runtime**: ~90 seconds on a standard modern CPU (Stage 1: ~50s, Stage 2 reranking of Top 1K: ~40s).

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
The hackathon enforces a **≤5 minute CPU-only constraint** on a standard machine. Generative LLMs (GPT-4, LLaMA, Mistral) require GPU and produce 1 token/second on CPU, making them impossible to use on 100K candidates. Our architecture uses only encoder-only models (`MiniLM`, `DistilRoBERTa`) which are extremely fast even on CPU.

### Why a Two-Stage Architecture?
Running Cross-Encoders on all 100K candidates sequentially on a CPU would take ~300 minutes — far beyond the constraint. By using the bi-encoder in Stage 1 to narrow to the Top 1,000 (top 1%), we apply the heavy Cross-Encoder and Zero-Shot models only where they matter, completing in 40 seconds.

### Why Rules + NLP, Not a Trained Model?
There is no labelled ground truth data for this specific JD. A rules engine derived directly from the JD text (skill taxonomy, title tiers, YoE bands) provides high-precision signals without needing training data.

### Graceful Degradation
If `spacy`, `sentence-transformers`, or `transformers` are not installed, the pipeline silently falls back — spaCy and MiniLM multipliers default to 1.0 (neutral), and Stage 2 is skipped. The base 8-component scoring always runs.

---

## 📦 Requirements

```
pandas
sentence-transformers
transformers
spacy
```

> Models used: `all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/nli-distilroberta-base`, `en_core_web_sm`
