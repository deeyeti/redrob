# Team Eclectic — Redrob India Runs Hackathon Submission

## Intelligent Candidate Discovery & Ranking System

**Challenge:** Redrob India Runs Data & AI Challenge
**Problem Statement:** Intelligent Candidate Discovery & Ranking for a Senior AI Engineer Role
**Team:** Eclectic

---

# Executive Summary

We developed a high-performance, explainable candidate ranking engine designed to identify and prioritize the most relevant candidates for a **Senior AI Engineer** position from a pool of **100,000 candidate profiles**.

The solution leverages a **multi-signal ranking architecture** that combines structured skill intelligence, experience evaluation, behavioral indicators, career progression analysis, company context, educational background, and candidate availability into a unified ranking framework.

Unlike traditional Applicant Tracking Systems (ATS) that rely heavily on keyword matching, our approach performs **context-aware candidate evaluation**, producing transparent rankings supported by evidence-based reasoning. The system is fully CPU-based, requires no external APIs or pre-trained models during inference, and processes the complete dataset within seconds while remaining compliant with all competition constraints.

---

# Problem Statement

The objective is to identify and rank the **Top 100 most relevant candidates** for a Senior AI Engineer position using a dataset containing over **100,000 candidate profiles**.

The ranking system must:

* Accurately understand job requirements.
* Evaluate candidate suitability across multiple dimensions.
* Generate explainable ranking decisions.
* Detect low-quality or suspicious candidate profiles.
* Operate under strict runtime and compute constraints.
* Produce recruiter-ready recommendations at scale.

Evaluation metrics include:

* NDCG@10 (50%)
* NDCG@50 (30%)
* MAP (15%)
* Precision@10 (5%)

---

# Solution Overview

Our solution implements a **JD-aware, multi-factor candidate ranking framework** that evaluates candidates using a weighted combination of technical, professional, and behavioral signals.

### Core Capabilities

* Intelligent Job Description understanding
* Multi-dimensional candidate assessment
* Explainable AI-based ranking
* Candidate quality validation
* High-throughput large-scale processing
* CPU-only execution with zero external dependencies

### Key Differentiators

* Goes beyond keyword-based candidate filtering.
* Evaluates real-world AI engineering experience and production impact.
* Incorporates recruiter engagement and hiring-readiness signals.
* Rewards demonstrated deployment and business impact.
* Produces transparent, recruiter-friendly explanations.
* Detects profile inconsistencies and suspicious candidate patterns.

---

# Ranking Architecture

The final ranking score is generated using a weighted ensemble of independent candidate signals.

| Component              | Weight | Purpose                                       |
| ---------------------- | ------ | --------------------------------------------- |
| Skill Match            | 35%    | Technical alignment with JD requirements      |
| Title & Role Relevance | 17%    | Role similarity and seniority alignment       |
| Experience Fit         | 13%    | Years-of-experience suitability               |
| Behavioral Signals     | 14%    | Recruiter engagement and hiring readiness     |
| Career Depth           | 7%     | Demonstrated impact and production experience |
| Company Context        | 6%     | Industry and organizational relevance         |
| Education              | 4%     | Academic pedigree and field relevance         |
| Availability           | 4%     | Joining readiness and location suitability    |

---

# Candidate Intelligence Framework

## Skill Match Engine (35%)

The skill matching layer evaluates candidates using:

* Handcrafted Senior AI Engineer skill taxonomy
* Pure-Python TF-IDF relevance scoring
* Skill synonym normalization
* Proficiency-weighted scoring
* Experience-duration weighting

The taxonomy includes critical AI competencies such as:

* Machine Learning
* Deep Learning
* NLP
* Transformers
* BERT
* Sentence Transformers
* Embeddings
* Retrieval Systems
* Vector Search
* Semantic Search
* RAG Pipelines
* LLM Applications
* Python Ecosystem

---

## Role Alignment Engine (17%)

Candidates are categorized using a role hierarchy:

### Tier 1 (Highest Relevance)

* AI Engineer
* Machine Learning Engineer
* NLP Engineer
* Applied Scientist
* AI Research Engineer

### Tier 2

* Software Engineer
* Backend Engineer
* Platform Engineer

### Tier 3

* Frontend Engineer
* QA Engineer
* Support Roles

### Tier 4

* Non-Technical Roles

This ensures preference for candidates whose career trajectory closely aligns with the target position.

---

## Experience Evaluation (13%)

The experience component evaluates candidate suitability against the JD's preferred experience range.

### Target Window

* Preferred Range: 5–9 Years
* Optimal Range: 6–8 Years

Candidates closest to the ideal experience band receive maximum scores, while scores gradually taper for underqualified or overqualified profiles.

---

## Behavioral Intelligence (14%)

Behavioral indicators provide insights into candidate responsiveness and hiring readiness.

Signals considered include:

* Recruiter Response Rate
* Interview Completion Rate
* GitHub Activity Score
* Profile Recency
* Recruiter Save Frequency
* Profile Completeness

These signals help identify candidates who are more likely to engage successfully during the hiring process.

---

## Career Depth Analysis (7%)

This component measures the practical depth of a candidate’s experience.

Signals include:

* AI/ML keyword diversity
* Quantified business impact detection
* Production deployment evidence
* Engineering ownership indicators
* Career description richness

Examples of detected evidence:

* "Improved latency by 20ms"
* "Scaled platform to 5M users"
* "Deployed recommendation system"
* "Conducted A/B testing"

---

## Company Context Evaluation (6%)

The system evaluates employer relevance using:

* Product vs Consulting company ratio
* AI-native organization preference
* SaaS, FinTech, EdTech, and Technology domain alignment
* Company scale suitability

Candidates from product-focused technology organizations receive higher relevance scores compared to consulting-heavy profiles.

---

## Education Assessment (4%)

Education is evaluated using:

### Institution Quality

* Tier 1: IITs, IISc, Top Global Universities
* Tier 2: Premier Engineering Institutions
* Tier 3: General Institutions

### Field Relevance

* Computer Science
* Artificial Intelligence
* Machine Learning
* Data Science
* Mathematics
* Statistics

---

## Availability Assessment (4%)

Candidate readiness is evaluated using:

* Notice Period
* Open-to-Work Status
* Geographic Alignment
* Work Mode Preference

Candidates available sooner and aligned with preferred hiring locations receive higher scores.

---

# Candidate Validation Framework

To improve ranking quality, a dedicated validation layer identifies suspicious or low-quality profiles.

## Hard Exclusion Rules

Profiles are automatically removed if:

* Employment dates are inconsistent
* Years of experience exceed realistic thresholds
* Multiple expert-level skills show zero practical duration

## Soft Penalty Rules

Profiles receive substantial score reductions if:

* Entire career history consists exclusively of large consulting organizations
* Limited evidence of product ownership or AI deployment exists

This improves precision while preserving potentially valuable edge-case candidates.

---

# Explainability Engine

Every recommendation is accompanied by a structured explanation containing:

* Current designation
* Total years of experience
* Top matched technical skills
* Industry and company context
* Production deployment evidence
* Recruiter engagement indicators
* Availability insights
* Potential concerns or risks

This ensures every ranking decision remains transparent, auditable, and recruiter-friendly.

---

# Performance & Scalability

The solution was specifically engineered for large-scale candidate evaluation.

### Processing Statistics

* Candidate Profiles Processed: 100,000+
* Ranking Output: Top 100 Candidates
* Runtime: ~10–15 Seconds
* Memory Usage: < 2 GB
* Compute Requirement: CPU Only
* External API Calls: None

### Competition Compliance

| Requirement    | Constraint  | Result          |
| -------------- | ----------- | --------------- |
| Runtime        | ≤ 5 Minutes | ~10–15 Seconds  |
| Memory         | ≤ 16 GB     | < 2 GB          |
| CPU Execution  | Required    | Achieved        |
| External Calls | Not Allowed | Fully Compliant |

---

# Key Design Decisions

### Why Not Use LLMs for Ranking?

Large Language Models introduce significant computational overhead and would violate runtime constraints when evaluating 100,000 profiles. Our structured scoring framework provides greater scalability, consistency, and reproducibility.

### Why Not Rely Solely on Text Similarity?

Candidate quality extends beyond textual similarity. Critical signals such as recruiter engagement, experience quality, availability, and production deployment history cannot be accurately captured using text matching alone.

### Why Analyze Career Descriptions Separately?

Many candidates mention advanced AI technologies and production systems within project descriptions without explicitly listing them as skills. Career-depth analysis captures this additional evidence and improves ranking accuracy.

### Why Penalize Consulting-Only Profiles?

The target role prioritizes hands-on product engineering and AI system ownership. Product-focused environments generally provide stronger alignment with these requirements, making company context an important ranking factor.

---

# Outcome

The final solution delivers:

* Accurate candidate-job matching
* Explainable ranking decisions
* Robust candidate quality validation
* Scalable processing for 100K+ profiles
* Full compliance with Redrob competition constraints

The system successfully transforms large-scale candidate datasets into recruiter-ready recommendations, enabling faster, more transparent, and higher-quality hiring decisions.
