"""
Team Eclectic — Intelligent Candidate Ranker (v3)
Redrob Hackathon: Senior AI Engineer Role

Usage:
    python rank.py --candidates candidates.jsonl --out team_eclectic.csv
    python rank.py --candidates sample_candidates.json --out team_eclectic.csv

Constraints: CPU-only, no network, ≤5 min, ≤16 GB RAM
"""

import os
import json
import argparse
import math
from datetime import date, datetime
from typing import Optional, Tuple, List, Dict
import pandas as pd
import re
import collections

# Optional NLP dependencies — gracefully disabled if not installed
try:
    import spacy as _spacy
    _NLP_MODEL = _spacy.load("en_core_web_sm", disable=["ner"])
except Exception:
    _NLP_MODEL = None

try:
    from sentence_transformers import SentenceTransformer as _ST, CrossEncoder as _CE
    from transformers import pipeline as _pipeline
    import numpy as _np
    _SMODEL = _ST("all-MiniLM-L6-v2")
    _CMODEL = _CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _ZMODEL = _pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
except Exception:
    _SMODEL = None
    _CMODEL = None
    _ZMODEL = None
    _np = None

# Ideal candidate description derived from the JD (used for MiniLM similarity)
_JD_IDEAL = (
    "Senior AI engineer with deep experience building production search and retrieval systems. "
    "Expert in vector databases, embeddings, semantic search, learning to rank, hybrid search BM25 dense, "
    "NLP, fine-tuning transformer models, deploying ML pipelines at scale, NDCG MRR evaluation."
)

# ============================================================
# JD SKILL TAXONOMY
# ============================================================

# Core JD skills (must-haves) — keyword → base_weight
JD_CORE_SKILLS = {
    # Embeddings / Retrieval
    "embeddings": 1.0,
    "sentence-transformers": 1.0,
    "sentence transformers": 1.0,
    "bert": 0.9,
    "transformers": 0.9,
    "word2vec": 0.7,
    "glove": 0.7,
    "fasttext": 0.7,
    "dense retrieval": 1.0,
    "semantic search": 1.0,
    "bi-encoder": 0.9,
    "cross-encoder": 0.9,
    "dpr": 0.9,
    "colbert": 0.9,
    "retrieval": 0.9,
    "information retrieval": 1.0,
    # Vector DBs / Search Infrastructure
    "faiss": 1.0,
    "pinecone": 1.0,
    "weaviate": 1.0,
    "qdrant": 1.0,
    "milvus": 1.0,
    "chroma": 0.8,
    "opensearch": 0.9,
    "elasticsearch": 0.9,
    "vector database": 1.0,
    "vector search": 1.0,
    "hybrid search": 1.0,
    "bm25": 0.9,
    "approximate nearest neighbor": 0.9,
    "ann": 0.8,
    "hnsw": 0.9,
    # Ranking & Evaluation
    "learning-to-rank": 1.0,
    "learning to rank": 1.0,
    "lambdamart": 1.0,
    "ranknet": 0.9,
    "ndcg": 0.9,
    "map": 0.7,
    "mrr": 0.8,
    "ranking": 0.8,
    "reranking": 0.9,
    "re-ranking": 0.9,
    "a/b testing": 0.8,
    # NLP / IR
    "nlp": 1.0,
    "natural language processing": 1.0,
    "natural language": 0.9,
    "text classification": 0.7,
    "named entity recognition": 0.7,
    "ner": 0.7,
    "question answering": 0.7,
    "information extraction": 0.8,
    # LLMs / Foundation Models
    "llm": 0.9,
    "large language model": 0.9,
    "gpt": 0.8,
    "llama": 0.8,
    "fine-tuning": 0.9,
    "fine tuning": 0.9,
    "lora": 0.9,
    "qlora": 0.9,
    "peft": 0.9,
    "rag": 0.9,
    "retrieval augmented generation": 1.0,
    "prompt engineering": 0.7,
    "instruction tuning": 0.8,
    # Python / ML Stack
    "python": 0.9,
    "pytorch": 0.9,
    "tensorflow": 0.8,
    "scikit-learn": 0.8,
    "sklearn": 0.8,
    "pandas": 0.6,
    "numpy": 0.6,
    "machine learning": 0.8,
    "deep learning": 0.8,
    "neural network": 0.8,
    "mlops": 0.8,
    "mlflow": 0.7,
    "model serving": 0.8,
}

# Nice-to-have skills — lower base weight
JD_NICE_SKILLS = {
    "xgboost": 0.7,
    "lightgbm": 0.7,
    "recommendation systems": 0.8,
    "recommendation": 0.7,
    "collaborative filtering": 0.7,
    "docker": 0.5,
    "kubernetes": 0.5,
    "fastapi": 0.6,
    "flask": 0.5,
    "distributed systems": 0.6,
    "kafka": 0.5,
    "spark": 0.6,
    "airflow": 0.5,
    "data pipelines": 0.5,
    "feature engineering": 0.7,
    "feature store": 0.7,
    "sql": 0.4,
    "rest api": 0.4,
    "api": 0.4,
    "git": 0.3,
    "aws": 0.4,
    "gcp": 0.4,
    "azure": 0.4,
}


SKILL_SYNONYMS = {
    "nlp": "natural language processing",
    "llm": "large language model",
    "llms": "large language model",
    "ann": "approximate nearest neighbor",
    "sentence-transformers": "sentence transformers",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "genai": "generative ai",
    "dnn": "deep neural network",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "cv": "computer vision",
    "ml": "machine learning",
    "ai": "artificial intelligence",
}

JD_TEXT = """
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.
Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.
Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.
Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.
Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
Strong Python. Yes really, we care about code quality.
Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.
LLM fine-tuning experience (LoRA, QLoRA, PEFT)
Experience with learning-to-rank models (XGBoost-based or neural)
Prior exposure to HR-tech, recruiting tech, or marketplace products
Background in distributed systems or large-scale inference optimization
Open-source contributions in the AI/ML space
"""

# Basic pre-computed TF-IDF for JD
def tokenize(text):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

JD_TOKENS = tokenize(JD_TEXT)
JD_TERM_FREQ = collections.Counter(JD_TOKENS)
JD_VOCAB_SIZE = len(JD_TERM_FREQ)

# ============================================================
# TITLE / ROLE TAXONOMY
# ============================================================

TITLE_TIERS = {
    # Tier 1 — direct JD match (1.0)
    1: [
        "ml engineer", "machine learning engineer", "ai engineer",
        "nlp engineer", "search engineer", "recommendation systems engineer",
        "ai research engineer", "ai researcher", "ai specialist",
        "data scientist", "research scientist", "applied scientist",
        "computer vision engineer",  # CV OK if they have NLP skills too
        "senior machine learning engineer", "staff machine learning engineer",
        "principal machine learning engineer", "lead machine learning engineer",
        "senior ai engineer", "senior data scientist",
        "senior nlp engineer", "senior software engineer (ml)",
        "machine learning scientist", "applied ml engineer",
        "ranking engineer", "relevance engineer",
        "junior ml engineer",  # still ML-track
        "junior machine learning engineer",
    ],
    # Tier 2 — adjacent (0.55)
    2: [
        "data engineer", "software engineer", "backend engineer",
        "full stack developer", "full stack engineer",
        "platform engineer", "infrastructure engineer",
        "backend developer", "python developer",
        "cloud engineer", "devops engineer",
    ],
    # Tier 3 — weak overlap (0.15)
    3: [
        "mobile developer", "frontend engineer", "frontend developer",
        "qa engineer", "quality assurance", "java developer",
        ".net developer", "ios developer", "android developer",
        "web developer", "product manager", "technical program manager",
    ],
    # Tier 0 — no overlap (0.0)
    0: [
        "hr manager", "human resources", "sales executive", "sales manager",
        "marketing manager", "content writer", "accountant", "civil engineer",
        "mechanical engineer", "business analyst", "customer support",
        "operations manager", "graphic designer", "project manager",
        "recruiter", "supply chain", "finance manager",
    ],
}

TITLE_SCORE_MAP = {1: 1.0, 2: 0.55, 3: 0.15, 0: 0.0}

# Consulting firms — reduce score when entire career is here
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "l&t infotech", "hexaware", "mindtree", "birlasoft", "mastech",
    "cyient", "niit technologies", "kpit", "zensar",
}

# Industries that signal product/tech company DNA
TECH_INDUSTRIES = {
    "technology", "software", "internet", "saas", "artificial intelligence",
    "machine learning", "data science", "fintech", "edtech", "healthtech",
    "e-commerce", "ecommerce", "cloud computing", "cybersecurity",
    "computer software", "information technology", "semiconductor",
    "telecommunications", "media technology", "analytics",
}

# Production/impact signals in career descriptions — prove real shipping
PRODUCTION_SIGNALS = [
    "production", "deployed", "shipped", "launched", "live system",
    "real-time", "real time", "at scale", "million users", "million requests",
    "billion", "low latency", "high throughput", "latency",
    "a/b test", "experiment", "online evaluation", "offline evaluation",
    "improved", "reduced", "increased", "optimized", "cut", "achieved",
    "metric", "recall", "precision", "ndcg", "mrr", "benchmark",
    "training pipeline", "inference", "model serving", "serving",
    "retrieval system", "search system", "ranking system", "recommendation",
]

CAVEAT_PATTERNS = [
    # Literal template phrases (from dataset generation)
    r"experimented with chatgpt",
    r"self-directed ml projects",
    r"transitioning toward more ai",
    r"learning modern ml practice",
    r"technical depth in ai is limited",
    r"lighter on technical depth",
    r"adjacent ml exposure",
    r"limited backend exposure",
    r"haven't done much application development",
    # Semantic variants — catching genuine hobbyists
    r"i(?:'m| am) (?:transitioning|pivoting|moving|exploring|learning|curious about)\b.{0,50}\b(?:ai|ml|machine learning|deep learning)",
    r"(?:no|limited|minimal|little|lack)\s+(?:formal|direct|hands.on)?\s*(?:ml|ai|machine learning|deep learning)\s+(?:experience|background|exposure)",
    r"(?:hobbyist|side project|passion project|pet project)\b.{0,60}\b(?:ai|ml|model|neural)",
    r"building (?:my|a) (?:foundation|fundamentals|basics)\b.{0,40}\b(?:ai|ml|machine)",
    r"(?:reading|taking|following)\s+(?:papers|books|courses|tutorials)\b.{0,40}\b(?:ai|ml|deep learning)",
    r"(?:looking to|want to|hope to|plan to)\b.{0,40}\b(?:break into|enter|move into|transition to)\b.{0,30}\b(?:ai|ml|machine learning)",
    r"(?:casually|occasionally|sometimes|rarely)\b.{0,40}\b(?:use|used|tried|played with)\b.{0,30}\b(?:ai|ml|model|gpt|llm)",
]

# Positive NLP patterns — reward genuine deep ML practitioners
POSITIVE_NLP_PATTERNS = [
    # Ownership verbs + retrieval/search/ranking objects
    r"(?:built|designed|architected|led|owned|developed|implemented|deployed|shipped|scaled)\b.{0,80}\b(?:retrieval|search|ranking|recommendation|embedding|pipeline|rag|llm|vector|index)",
    # Quantified scale claims
    r"\d+[\s]?[kmb](?:illion)?\+?\s*(?:queries|requests|documents|users|records|items)\b",
    # Quantified metric impact
    r"(?:improved|reduced|increased|boosted|cut|achieved|drove|lifted)\b.{0,60}\b(?:latency|ndcg|mrr|recall|precision|throughput|accuracy|p95|p99)\b.{0,40}\b(?:\d+[%x×]|percentage|points)",
    # Architecture specifics
    r"(?:hybrid|dense|sparse)\s+(?:retrieval|search|index|re.?rank)",
    r"(?:fine.tun|lora|qlora|peft|adapter)\b.{0,60}\b(?:model|bert|llm|transformer)",
    r"(?:cross.encoder|bi.encoder|rerank|re-rank|two.stage|multi.stage)\b",
    r"(?:bm25|tfidf|tf.idf)\b.{0,50}\b(?:dense|embedding|vector|neural)",
    r"(?:faiss|qdrant|weaviate|milvus|pinecone|opensearch|elasticsearch)\b.{0,60}\b(?:index|deploy|produc|build|migrat)",
    # Rigorous offline/online evaluation
    r"(?:a/b\s+test|experiment|holdout|eval(?:uation)?)\b.{0,60}\b(?:ndcg|mrr|recall|precision|online|offline)",
    r"(?:designed|built|ran|automated)\b.{0,60}\b(?:eval(?:uation)?\s+(?:framework|harness|pipeline|suite))",
    # LLM production work
    r"(?:llm|gpt|claude|mistral|llama)\b.{0,60}\b(?:produc|deploy|serve|latency|throughput|cost|inference)",
    r"(?:prompt\s+engineering|chain.of.thought|rag\s+pipeline|grounding|hallucination)\b",
]


# Keywords in career descriptions proving ML/AI depth (subset of JD core)
CAREER_ML_KEYWORDS = [
    "embedding", "embeddings", "vector", "retrieval", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "transformer", "bert", "sentence-transformer", "sentence transformer",
    "nlp", "natural language", "llm", "rag", "fine-tun", "pytorch",
    "tensorflow", "learning to rank", "learning-to-rank", "ranking",
    "semantic search", "hybrid search", "dense retrieval", "bm25",
    "recommendation", "collaborative filtering", "neural network",
    "deep learning", "machine learning", "xgboost", "lightgbm",
    "mlops", "model", "feature engineering", "feature store",
]

TODAY = date.today()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def days_since(date_str: Optional[str]) -> int:
    """Days since a date string (YYYY-MM-DD). Returns 9999 if invalid."""
    if not date_str:
        return 9999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999


def normalize(value: float, lo: float, hi: float) -> float:
    """Clamp-normalize value to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def log_scale(value: float, cap: float = 50.0) -> float:
    """Log-normalize (e.g. for count signals)."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(cap))


def get_title_tier(title: str) -> int:
    """Return tier (0-3) for a job title string."""
    t = title.lower().strip()
    for tier, keywords in TITLE_TIERS.items():
        for kw in keywords:
            if kw in t:
                return tier
    # Heuristic fallbacks
    if any(x in t for x in ["engineer", "developer", "scientist", "analyst"]):
        return 2
    return 3


# ============================================================
# SCORING COMPONENTS
# ============================================================

def score_skills(candidate: dict) -> Tuple[float, List[str]]:
    """
    Returns (skill_score 0-1, list of matched core skill names for reasoning).
    Looks at skills[], career descriptions, profile summary, and assessment scores.
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})

    # Build a text blob for contextual keyword matching
    text_blob = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(j.get("description", "") for j in career),
        " ".join(j.get("title", "") for j in career),
    ]).lower()

    # Normalize skill names from candidate profile
    skill_map = {}  # name_lower -> {proficiency, duration_months, endorsements}
    for sk in skills:
        name_lower = sk.get("name", "").lower()
        name_lower = SKILL_SYNONYMS.get(name_lower, name_lower)
        skill_map[name_lower] = sk

    proficiency_weight = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}

    matched_core = []
    core_score = 0.0
    core_max = 0.0
    nice_score = 0.0
    nice_max = 0.0

    def skill_weight(keyword, base_weight, in_skills=True):
        sk_data = skill_map.get(keyword)
        if sk_data:
            prof = proficiency_weight.get(sk_data.get("proficiency", "beginner"), 0.4)
            dur = min(sk_data.get("duration_months", 6) / 36.0, 1.0)
            weight = base_weight * (0.5 * prof + 0.5 * dur)
        elif keyword in text_blob:
            # Found in career/summary text — partial credit
            weight = base_weight * 0.4
        else:
            weight = 0.0
        # Bonus if they have an assessment score for this skill
        for asm_key, asm_val in assessment_scores.items():
            if keyword in asm_key.lower():
                weight = min(weight + base_weight * 0.1 * (asm_val / 100), base_weight)
        return weight

    for keyword, base in JD_CORE_SKILLS.items():
        w = skill_weight(keyword, base)
        core_max += base
        if w > 0:
            core_score += w
            # Collect for reasoning (only unique, significant matches)
            if w >= base * 0.5:
                sk_data = skill_map.get(keyword)
                if sk_data:
                    matched_core.append((sk_data.get("name", keyword), w))
                elif keyword in text_blob:
                    matched_core.append((keyword, w))

    for keyword, base in JD_NICE_SKILLS.items():
        w = skill_weight(keyword, base)
        nice_max += base
        if w > 0:
            nice_score += w

    # Normalize separately then combine
    core_norm = core_score / core_max if core_max > 0 else 0.0
    nice_norm = nice_score / nice_max if nice_max > 0 else 0.0

    # TF-IDF calculation against JD text
    cand_tokens = tokenize(text_blob)
    cand_tf = collections.Counter(cand_tokens)
    
    tfidf_score = 0.0
    for term, freq in cand_tf.items():
        if term in JD_TERM_FREQ:
            # Simple TF-IDF: tf * log(1 + jd_freq)
            tfidf_score += freq * math.log1p(JD_TERM_FREQ[term])
    
    # Cap TF-IDF bonus
    tfidf_norm = min(1.0, tfidf_score / 200.0)

    final = 0.60 * core_norm + 0.15 * nice_norm + 0.25 * tfidf_norm

    # Sort matched core skills by weight desc, deduplicate
    matched_core.sort(key=lambda x: -x[1])
    seen = set()
    deduped = []
    for name, w in matched_core:
        if name.lower() not in seen:
            seen.add(name.lower())
            deduped.append(name)
        if len(deduped) >= 5:
            break

    return final, deduped


def score_title(candidate: dict) -> Tuple[float, int]:
    """Returns (title_score 0-1, tier) based on current title and career history."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_tier = get_title_tier(profile.get("current_title", ""))

    # Also look at career history — if ever had ML/AI role, bump up
    best_past_tier = 3
    for job in career:
        t = get_title_tier(job.get("title", ""))
        best_past_tier = min(best_past_tier, t)  # lower tier number = better

    # Current title dominates; past tier can help if current is adjacent
    if current_tier == 1 or best_past_tier == 1:
        effective_tier = 1
    elif current_tier == 2 and best_past_tier <= 2:
        effective_tier = 2
    else:
        effective_tier = max(current_tier, best_past_tier)

    return TITLE_SCORE_MAP[effective_tier], effective_tier


def score_yoe(candidate: dict) -> float:
    """Returns 0-1 score based on years_of_experience against JD requirements."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if 6 <= yoe <= 8:
        return 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        return 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        return 0.55
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        return 0.30
    else:
        return 0.10


def score_behavioral(candidate: dict) -> Tuple[float, dict]:
    """
    Returns (behavioral_score 0-1, breakdown dict for reasoning).
    Uses 6 platform signals.
    """
    sig = candidate.get("redrob_signals", {})

    response_rate = sig.get("recruiter_response_rate", 0.0)  # 0-1
    interview_rate = sig.get("interview_completion_rate", 0.0)  # 0-1
    github_raw = sig.get("github_activity_score", -1)  # 0-100 or -1
    last_active_days = days_since(sig.get("last_active_date"))
    saved_30d = sig.get("saved_by_recruiters_30d", 0)
    completeness = sig.get("profile_completeness_score", 0) / 100.0

    # GitHub: -1 means no GitHub; treat as 0 for scoring
    github_score = normalize(max(github_raw, 0), 0, 100)

    # Recency of last activity (0 = active today, 1 = active 365+ days ago)
    recency_score = 1.0 - normalize(last_active_days, 0, 365)

    saved_score = log_scale(saved_30d, cap=20)

    combined = (
        0.35 * response_rate
        + 0.25 * interview_rate
        + 0.15 * github_score
        + 0.10 * recency_score
        + 0.10 * saved_score
        + 0.05 * completeness
    )

    breakdown = {
        "response_rate": response_rate,
        "interview_rate": interview_rate,
        "github": github_raw,
        "last_active_days": last_active_days,
        "saved_30d": saved_30d,
    }
    return combined, breakdown


def score_education(candidate: dict) -> float:
    """Returns 0-1 based on best education tier and field relevance."""
    education = candidate.get("education", [])
    if not education:
        return 0.3

    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.3}
    relevant_fields = {
        "computer science", "cs", "software engineering", "artificial intelligence",
        "machine learning", "data science", "mathematics", "statistics",
        "computational linguistics", "information technology", "electronics",
        "electrical engineering", "information systems",
    }

    best_tier_score = 0.0
    field_bonus = 0.0
    for edu in education:
        t = tier_scores.get(edu.get("tier", "unknown"), 0.3)
        if t > best_tier_score:
            best_tier_score = t
        field = edu.get("field_of_study", "").lower()
        if any(rf in field for rf in relevant_fields):
            field_bonus = 0.1  # one-time bonus

    return min(1.0, best_tier_score + field_bonus)


def score_career_depth(candidate: dict) -> Tuple[float, dict]:
    """
    Scores the quality of evidence in career descriptions.
    Two dimensions:
      1. ML/AI keyword breadth in career descriptions (unique domain terms found)
      2. Production deployment signals (proof of real shipping)
    Returns (career_depth_score 0-1, info dict for reasoning).
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Concatenate all career description text + headline + summary
    all_text = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(j.get("description", "") for j in career),
    ]).lower()

    # 1. Count unique ML/AI domain keywords found in career text
    found_ml_kws = sum(1 for kw in CAREER_ML_KEYWORDS if kw in all_text)
    # Normalize: 10+ unique terms = full score (saturates at 10)
    ml_breadth_score = min(found_ml_kws / 10.0, 1.0)

    # 2. Count production/impact signals
    found_prod = sum(1 for sig in PRODUCTION_SIGNALS if sig in all_text)
    
    # 2b. Quantified impact detection (e.g. 50%, 10x, 20ms, 5 million)
    quant_patterns = [
        r'\b\d+\s*%', r'\b\d+\s*x\b', r'\b\d+\s*ms\b', 
        r'\b\d+\s*(?:million|billion|k|m|b)\b', r'\$\d+'
    ]
    found_quant = sum(1 for p in quant_patterns if re.search(p, all_text))
    
    # Normalize: 6+ production signals + 2+ quant signals = full score
    prod_score = min((found_prod + found_quant) / 8.0, 1.0)

    # 3. Career description quality — prefer longer, richer descriptions
    avg_desc_len = 0.0
    if career:
        avg_desc_len = sum(len(j.get("description", "")) for j in career) / len(career)
    # 300+ chars avg = full score (rich descriptions)
    desc_quality = min(avg_desc_len / 300.0, 1.0)

    combined = (
        0.45 * ml_breadth_score
        + 0.40 * prod_score
        + 0.15 * desc_quality
    )

    info = {
        "ml_keywords_found": found_ml_kws,
        "production_signals": found_prod,
    }
    return combined, info


def score_company_context(candidate: dict) -> Tuple[float, dict]:
    """
    Scores the quality of company/industry context in career history.
    Rewards: product/tech companies, growth trajectory, relevant industry.
    Penalises: pure consulting career (already handled in disqualifier, but
               here we give a graded penalty rather than binary exclude).
    Returns (company_context_score 0-1, info dict for reasoning).
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    if not career:
        return 0.3, {"product_ratio": 0.0, "best_industry": "unknown"}

    # Classify each job as consulting / tech-product / other
    consulting_count = 0
    tech_product_count = 0
    best_company_size_score = 0.0
    best_industry = "unknown"

    # Company size score: startups/scaleups preferred by JD
    size_score_map = {
        "1-10": 0.7,       # very early stage — risky
        "11-50": 0.9,      # startup
        "51-200": 1.0,     # scaleup (sweet spot)
        "201-500": 0.95,
        "501-1000": 0.85,
        "1001-5000": 0.75,
        "5001-10000": 0.65,
        "10001+": 0.6,     # big corp — can still be great (but JD prefers product)
    }

    for job in career:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()
        size = job.get("company_size", "")

        is_consulting = any(firm in company for firm in CONSULTING_FIRMS)
        is_tech = any(tech in industry for tech in TECH_INDUSTRIES)

        if is_consulting:
            consulting_count += 1
        elif is_tech:
            tech_product_count += 1

        if is_tech and industry != "unknown":
            best_industry = job.get("industry", best_industry)

        sz = size_score_map.get(size, 0.5)
        if not is_consulting and sz > best_company_size_score:
            best_company_size_score = sz

    total = len(career)
    consulting_ratio = consulting_count / total
    tech_product_ratio = tech_product_count / total

    # Product/tech company score
    product_score = tech_product_ratio  # 0-1

    # Consulting penalty: 0=no consulting (great), 1=all consulting (bad)
    consulting_penalty = 1.0 - (consulting_ratio * 0.7)  # max 0.7 penalty

    # Also factor in current industry from profile
    current_industry = profile.get("current_industry", "").lower()
    industry_bonus = 0.15 if any(tech in current_industry for tech in TECH_INDUSTRIES) else 0.0

    combined = (
        0.45 * product_score
        + 0.30 * consulting_penalty
        + 0.15 * best_company_size_score
        + 0.10 * min(industry_bonus / 0.15, 1.0)  # normalize bonus
    )
    combined = min(1.0, combined)

    info = {
        "product_ratio": round(tech_product_ratio, 2),
        "consulting_ratio": round(consulting_ratio, 2),
        "best_industry": best_industry,
    }
    return combined, info


def score_availability(candidate: dict) -> Tuple[float, dict]:
    """Returns (availability_score 0-1, info dict for reasoning)."""
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    open_to_work = sig.get("open_to_work_flag", False)
    notice_days = sig.get("notice_period_days", 90)
    willing_relocate = sig.get("willing_to_relocate", False)
    location = profile.get("location", "").lower()
    preferred_mode = sig.get("preferred_work_mode", "")

    # Notice period scoring
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.35
    else:
        notice_score = 0.1

    # Location signal
    preferred_locations = ["pune", "noida", "delhi", "ncr", "gurugram", "gurgaon",
                            "hyderabad", "mumbai", "bangalore", "bengaluru"]
    loc_score = 0.5  # neutral default
    if any(city in location for city in ["pune", "noida"]):
        loc_score = 1.0
    elif any(city in location for city in preferred_locations):
        loc_score = 0.75
    elif willing_relocate:
        loc_score = 0.6

    otw_bonus = 0.3 if open_to_work else 0.0
    reloc_bonus = 0.1 if willing_relocate else 0.0
    mode_bonus = 0.1 if preferred_mode in ("hybrid", "flexible") else 0.0

    combined = (
        0.4 * notice_score
        + 0.3 * loc_score
        + otw_bonus
        + reloc_bonus
        + mode_bonus
    )
    combined = min(1.0, combined)

    info = {
        "open_to_work": open_to_work,
        "notice_days": notice_days,
        "location": profile.get("location", ""),
        "willing_relocate": willing_relocate,
        "preferred_mode": preferred_mode,
    }
    return combined, info


def _extract_verb_objects(text: str) -> Dict[str, List[str]]:
    """Use spaCy to extract (verb → [objects]) from text (first 500 chars).
    Returns dict keyed by verb lemma. Only called when _NLP_MODEL is available."""
    out: Dict[str, List[str]] = collections.defaultdict(list)
    doc = _NLP_MODEL(text[:500])
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl", "ccomp"):
            objs = [
                child.lemma_.lower() for child in token.children
                if child.dep_ in ("dobj", "attr", "pobj", "nsubjpass") and len(child.text) > 2
            ]
            if objs:
                out[token.lemma_.lower()].extend(objs)
    return dict(out)


# Strong ownership verbs — subject built/shipped something
_STRONG_VERBS = frozenset([
    "build", "design", "architect", "develop", "implement", "deploy",
    "ship", "scale", "own", "lead", "drive", "launch", "migrate",
    "create", "write", "train", "fine-tune", "optimize",
])
# Weak verbs — subject is only familiar with something
_WEAK_VERBS = frozenset([
    "know", "learn", "explore", "experiment", "try", "play",
    "familiarize", "study", "read", "understand", "expose",
])
# JD-relevant object nouns
_JD_NOUNS = frozenset([
    "pipeline", "system", "model", "retrieval", "search", "index", "embedding",
    "ranker", "ranking", "recommendation", "vector", "encoder", "llm",
    "transformer", "rag", "evaluation", "experiment", "service", "api",
])


def score_nlp_context(candidate: dict) -> float:
    """Returns a multiplier: 0.05 (near-exclude) … 1.3 (strong positive NLP signal)."""
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    # Check if entire career is at consulting firms only
    if career and len(career) >= 2:
        all_consulting = all(
            any(firm in job.get("company", "").lower() for firm in CONSULTING_FIRMS)
            for job in career
        )
        if all_consulting:
            return 0.1

    # Build full text blob (skills + titles + headline + summary + descriptions)
    skill_names = " ".join(sk.get("name", "").lower() for sk in skills)
    career_titles = " ".join(job.get("title", "").lower() for job in career)
    career_descs = " ".join(job.get("description", "").lower() for job in career)
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    full_text = " ".join([skill_names, career_titles, headline, summary, career_descs])

    # Count how many genuine JD core ML skills this candidate has in their skills section
    skill_names_list = [sk.get("name", "").lower() for sk in skills]
    strong_ml_skill_count = sum(
        1 for s in skill_names_list
        if s in JD_CORE_SKILLS or s in JD_NICE_SKILLS
    )
    has_strong_ml_skills = strong_ml_skill_count >= 2

    # --- Layer 1: Caveat detection (penalty) ---
    # Only penalize if the candidate ALSO has weak ML skills.
    # A genuine Senior NLP Engineer can mention a past ChatGPT experiment without being penalized.
    if not has_strong_ml_skills:
        for pat in CAVEAT_PATTERNS:
            if re.search(pat, full_text):
                return 0.3

    multiplier = 1.0

    # --- Layer 1: Positive regex boosting ---
    positive_hits = sum(1 for pat in POSITIVE_NLP_PATTERNS if re.search(pat, full_text))
    if positive_hits >= 3:
        multiplier *= 1.08  # max +8% bonus for multiple strong signals
    elif positive_hits >= 1:
        multiplier *= 1.04  # +4% for any positive signal

    # --- Layer 2: spaCy verb-object extraction ---
    if _NLP_MODEL is not None:
        combined_text = (summary + " " + career_descs)[:3000]
        vo = _extract_verb_objects(combined_text)
        strong_jd_hits = sum(
            1 for v, objs in vo.items()
            if v in _STRONG_VERBS and any(o in _JD_NOUNS for o in objs)
        )
        weak_jd_hits = sum(
            1 for v, objs in vo.items()
            if v in _WEAK_VERBS and any(o in _JD_NOUNS for o in objs)
        )
        if strong_jd_hits >= 2:
            multiplier *= 1.05  # +5% for multiple strong ownership verbs
        elif strong_jd_hits == 1:
            multiplier *= 1.02  # +2% for one
        if weak_jd_hits >= 2 and strong_jd_hits == 0:
            multiplier *= 0.95  # -5% if only "familiar with" verbs, no ownership

    # Check current title tier
    current_tier = get_title_tier(profile.get("current_title", ""))
    best_career_tier = min(
        (get_title_tier(job.get("title", "")) for job in career),
        default=3
    )
    # Hard non-tech: tier 0 current title AND no ML signal anywhere
    ml_keywords = [
        "machine learning", "deep learning", "nlp", "neural", "embedding",
        "transformer", "llm", "retrieval", "vector", "ranking", "ml engineer",
    ]
    has_ml_signal = any(kw in full_text for kw in ml_keywords)
    if current_tier == 0 and best_career_tier >= 2 and not has_ml_signal:
        return 0.05

    return min(multiplier, 1.15)  # Cap at +15% total NLP bonus


# ============================================================
# REASONING GENERATION
# ============================================================

def generate_reasoning(candidate: dict, yoe: float, title: str,
                        matched_skills: list, behavioral: dict,
                        avail_info: dict, yoe_score: float,
                        skill_score: float, title_tier: int,
                        depth_info: dict, company_info: dict) -> str:
    """Generate a specific, honest, JD-connected reasoning string."""
    parts = []

    # 1. Title + YoE
    parts.append(f"{title} with {yoe:.1f} yrs exp")

    # 2. Top matched skills
    if matched_skills:
        top_skills = matched_skills[:3]
        parts.append(f"skills: {', '.join(top_skills)}")
    else:
        parts.append("limited JD-matched skills")

    # 3. Company/industry context
    prod_ratio = company_info.get("product_ratio", 0.0)
    consult_ratio = company_info.get("consulting_ratio", 0.0)
    industry = company_info.get("best_industry", "")
    if prod_ratio >= 0.6:
        ind_str = f" in {industry}" if industry and industry != "unknown" else ""
        parts.append(f"product-company background{ind_str}")
    elif consult_ratio >= 0.5:
        parts.append(f"predominantly consulting background ({int(consult_ratio*100)}%)")

    # 4. Production depth signal
    prod_signals = depth_info.get("production_signals", 0)
    ml_kws = depth_info.get("ml_keywords_found", 0)
    if prod_signals >= 3:
        parts.append("strong production deployment evidence")
    elif ml_kws >= 5:
        parts.append(f"solid ML domain depth ({ml_kws} JD concepts in career text)")

    # 5. Key behavioral signals
    rr = behavioral.get("response_rate", 0)
    ir = behavioral.get("interview_rate", 0)
    gh = behavioral.get("github", -1)

    if rr >= 0.8:
        parts.append(f"excellent recruiter response rate ({int(rr*100)}%)")
    elif rr >= 0.5:
        parts.append(f"good response rate ({int(rr*100)}%)")
    else:
        parts.append(f"low response rate ({int(rr*100)}%) — engagement risk")

    if gh >= 60:
        parts.append(f"strong GitHub activity ({gh}/100)")
    elif gh == -1:
        parts.append("no GitHub linked")

    # 6. Location + availability
    loc = avail_info.get("location", "")
    notice = avail_info.get("notice_days", 90)
    otw = avail_info.get("open_to_work", False)

    if loc:
        parts.append(f"based in {loc}")
    if otw:
        parts.append("actively open to work")
    if notice <= 30:
        parts.append("notice ≤30 days")
    elif notice > 90:
        parts.append(f"long notice period ({notice}d)")

    # 7. Honest concerns (only flag genuine problems)
    concerns = []
    # Only flag skill weakness for non-ML-titled candidates — tier-1 titles prove the track
    if skill_score < 0.15 and title_tier >= 2:
        concerns.append("limited verified JD skill evidence")
    if title_tier >= 3:
        concerns.append("non-ML background")
    if yoe_score < 0.4:
        yoe_val = candidate.get("profile", {}).get("years_of_experience", 0)
        concerns.append(f"YoE {yoe_val:.1f} outside 5-9yr band")
    if behavioral.get("interview_rate", 1) < 0.4:
        concerns.append(f"low interview completion ({int(ir*100)}%)")
    if consult_ratio >= 0.8:
        concerns.append("career mostly at consulting firms")

    if concerns:
        parts.append("Concern: " + "; ".join(concerns))

    return ". ".join(parts) + "."


# ============================================================
# MAIN SCORING FUNCTION
# ============================================================

def score_candidate(candidate: dict, jd_sim: float = 0.5) -> Optional[dict]:
    """Score a single candidate. jd_sim is MiniLM cosine sim to ideal JD (0-1)."""

    candidate_id = candidate.get("candidate_id")
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "Unknown")

    # Step 2: Disqualifier + NLP context multiplier
    disqualifier_mult = score_nlp_context(candidate)

    # Step 2b: MiniLM semantic similarity multiplier (Layer 3)
    # MiniLM cosine sims have a ~0.3 baseline even for unrelated texts.
    # Anchored at sim=0.45 (neutral SWE), range kept tight [0.93, 1.07]
    # to fine-tune without overriding base scoring differentiation:
    #   sim=0.20 (irrelevant) -> 0.93x slight penalty
    #   sim=0.33 (HR manager) -> 0.93x slight penalty
    #   sim=0.45 (neutral SWE) -> 1.00x neutral
    #   sim=0.55 (ML engineer) -> 1.04x slight bonus
    #   sim=0.75+ (perfect match) -> 1.07x (capped)
    if _SMODEL is not None:
        jd_sim_mult = 1.0 + 0.7 * (jd_sim - 0.45)
        jd_sim_mult = max(0.93, min(1.07, jd_sim_mult))
    else:
        jd_sim_mult = 1.0

    # Step 3: Score components
    skill_score, matched_skills = score_skills(candidate)
    title_score, title_tier = score_title(candidate)
    yoe_sc = score_yoe(candidate)
    behavioral_sc, behavioral_info = score_behavioral(candidate)
    education_sc = score_education(candidate)
    availability_sc, avail_info = score_availability(candidate)
    career_depth_sc, depth_info = score_career_depth(candidate)
    company_ctx_sc, company_info = score_company_context(candidate)

    # Step 4: Weighted combination
    # Weights: skill(0.35) + title(0.17) + yoe(0.13) + behavioral(0.14)
    #          + career_depth(0.07) + company_ctx(0.06) + education(0.04) + availability(0.04) = 1.00
    raw_score = (
        0.35 * skill_score
        + 0.17 * title_score
        + 0.13 * yoe_sc
        + 0.14 * behavioral_sc
        + 0.07 * career_depth_sc
        + 0.06 * company_ctx_sc
        + 0.04 * education_sc
        + 0.04 * availability_sc
    )

    final_score = round(max(0.0, raw_score * disqualifier_mult * jd_sim_mult), 4)

    # Step 5: Rich reasoning
    reasoning = generate_reasoning(
        candidate=candidate,
        yoe=yoe,
        title=current_title,
        matched_skills=matched_skills,
        behavioral=behavioral_info,
        avail_info=avail_info,
        yoe_score=yoe_sc,
        skill_score=skill_score,
        title_tier=title_tier,
        depth_info=depth_info,
        company_info=company_info,
    )

    return {
        "candidate_id": candidate_id,
        "score": final_score,
        "reasoning": reasoning,
        # Debug fields (stripped from final output)
        "_skill": round(skill_score, 4),
        "_title": round(title_score, 4),
        "_yoe": round(yoe_sc, 4),
        "_behavioral": round(behavioral_sc, 4),
        "_career_depth": round(career_depth_sc, 4),
        "_company_ctx": round(company_ctx_sc, 4),
        "_education": round(education_sc, 4),
        "_availability": round(availability_sc, 4),
        "_disqualifier": disqualifier_mult,
    }


# ============================================================
# PIPELINE
# ============================================================

def open_file(path: str):
    """Open either .jsonl (plain), .json (array), or .jsonl.gz."""
    import gzip
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_candidates(path: str):
    """Yield individual candidate dicts from JSONL or JSON array."""
    print(f"Loading candidates from: {path}")

    with open_file(path) as f:
        content = f.read()

    # Try JSON array first (sample_candidates.json format)
    stripped = content.strip()
    if stripped.startswith("[") or stripped.startswith("{\"candidates\""):
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                yield from data
                return
            elif isinstance(data, dict) and "candidates" in data:
                yield from data["candidates"]
                return
        except json.JSONDecodeError:
            pass

    # Fall back to JSONL (one JSON object per line)
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _build_jd_sim_map(candidates_path: str) -> Dict[str, float]:
    """Layer 3: Batch-encode all summaries with MiniLM, return {candidate_id: cosine_sim}."""
    if _SMODEL is None or _np is None:
        return {}

    print("  [NLP Layer 3] Loading summaries for MiniLM encoding...")
    ids, texts = [], []
    for c in load_candidates(candidates_path):
        cid = c.get("candidate_id", "")
        summary = c.get("profile", {}).get("summary", "") or ""
        desc = " ".join(j.get("description", "") for j in c.get("career_history", []))[:300]
        ids.append(cid)
        texts.append((summary + " " + desc)[:512])

    jd_vec = _SMODEL.encode(_JD_IDEAL, convert_to_numpy=True, normalize_embeddings=True)
    print(f"  [NLP Layer 3] Encoding {len(texts):,} summaries in batches...")
    cand_vecs = _SMODEL.encode(
        texts,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sims = (cand_vecs @ jd_vec).tolist()  # cosine sim (vecs are normalized)
    print("  [NLP Layer 3] Done.")
    return dict(zip(ids, sims))


def run_pipeline(candidates_path: str, output_path: str, debug: bool = False):
    # Layer 3 pre-pass: batch MiniLM encoding (skipped gracefully if not installed)
    jd_sim_map = _build_jd_sim_map(candidates_path)
    if jd_sim_map:
        print(f"  [NLP Layer 3] JD similarity computed for {len(jd_sim_map):,} candidates.")

    # Statistics
    total = 0
    disqualified_count = 0
    results = []
    print(f"Loading candidates from: {candidates_path}")
    for candidate in load_candidates(candidates_path):
        total += 1
        cid = candidate.get("candidate_id", "")
        jd_sim = jd_sim_map.get(cid, 0.5)  # default neutral if not found
        result = score_candidate(candidate, jd_sim=jd_sim)
        if result:
            results.append(result)
            if "reasoning" in result and result["score"] < 0.1:
                disqualified_count += 1
        if total % 10000 == 0:
            print(f"  Processed {total:,} candidates... ({len(results):,} scored)")

    print("\nProcessing complete:")
    print(f"  Total candidates: {total:,}")
    print(f"  Soft disqualified: {disqualified_count:,}")
    print(f"  Valid candidates: {len(results):,}")

    # Scores are left as raw values without normalization
    for r in results:
        r["score"] = round(r["score"], 4)

    # Sort: primary by score desc, secondary by candidate_id asc (tie-break rule)
    df = pd.DataFrame(results)
    df = df.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # ============================================================
    # NLP STAGE 2: Cross-Encoder & Implicit Skill Extraction
    # ============================================================
    n_stage2 = min(1000, len(df))
    if n_stage2 > 0 and _CMODEL is not None and _ZMODEL is not None:
        print(f"  [NLP Stage 2] Extracting full profiles for Top {n_stage2} candidates...")
        stage2_cids = set(df.head(n_stage2)["candidate_id"])
        stage2_candidates = {}
        for candidate in load_candidates(candidates_path):
            cid = candidate.get("candidate_id", "")
            if cid in stage2_cids:
                stage2_candidates[cid] = candidate
                if len(stage2_candidates) == n_stage2:
                    break
        
        pairs = []
        texts = []
        cids = []
        for cid in df.head(n_stage2)["candidate_id"]:
            c = stage2_candidates[cid]
            summary = c.get("profile", {}).get("summary", "") or ""
            desc = " ".join(j.get("description", "") for j in c.get("career_history", []))[:1000]
            text = (summary + " " + desc)[:1500]
            pairs.append((_JD_IDEAL, text))
            texts.append(text)
            cids.append(cid)
            
        print("  [NLP Stage 2] Running Cross-Encoder (ms-marco-MiniLM)...")
        ce_scores = _CMODEL.predict(pairs, batch_size=32, show_progress_bar=False)
        import math
        def expit(x): return 1 / (1 + math.exp(-x))
        ce_mults = [0.8 + 0.4 * expit(score) for score in ce_scores] # 0.8x to 1.2x
        
        print("  [NLP Stage 2] Running Zero-Shot Classification for Implicit Skills...")
        implicit_skills = ["Vector Databases", "LLM Fine-tuning", "Learning to Rank", "Semantic Search"]
        # Using distilroberta for speed
        zs_results = _ZMODEL(texts, implicit_skills, multi_label=True, batch_size=32)
        
        print("  [NLP Stage 2] Re-ranking...")
        new_scores = []
        new_reasonings = []
        for i, cid in enumerate(cids):
            base_score = df.loc[i, "score"]
            base_reasoning = df.loc[i, "reasoning"]
            
            ce_mult = ce_mults[i]
            zs_res = zs_results[i]
            
            found_implicit = []
            for label, p in zip(zs_res["labels"], zs_res["scores"]):
                if p > 0.6:
                    found_implicit.append(label)
                    
            zs_mult = 1.0 + (0.05 * len(found_implicit)) # up to +20%
            final_score = round(base_score * ce_mult * zs_mult, 4)
            
            reasoning = base_reasoning
            if found_implicit:
                reasoning = reasoning.rstrip(".") + f". implicitly detected skills: {', '.join(found_implicit)}."
                
            new_scores.append(final_score)
            new_reasonings.append(reasoning)
            
        df.loc[:n_stage2-1, "score"] = new_scores
        df.loc[:n_stage2-1, "reasoning"] = new_reasonings
        
        # Re-sort top candidates based on new final scores
        df_stage2 = df.head(n_stage2).copy()
        df_stage2 = df_stage2.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
        df = pd.concat([df_stage2, df.iloc[n_stage2:]]).reset_index(drop=True)
        print("  [NLP Stage 2] Complete.")

    # Top 100 only (or fewer if sample dataset)
    n_top = min(100, len(df))
    df_top = df.head(n_top).copy()
    df_top["rank"] = range(1, n_top + 1)

    print("\nTop 10 candidates:")
    for _, row in df_top.head(10).iterrows():
        print(f"  Rank {int(row['rank'])}: {row['candidate_id']} | score={row['score']:.4f} | {row['reasoning'][:80]}...")

    # Output columns
    output_cols = ["candidate_id", "rank", "score", "reasoning"]
    if debug:
        output_cols += ["_skill", "_title", "_yoe", "_behavioral", "_career_depth", "_company_ctx", "_education", "_availability", "_disqualifier"]

    final_output = df_top[output_cols]
    final_output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df_top)} rows to: {output_path}")

    return df_top


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team Eclectic — Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or sample_candidates.json")
    parser.add_argument("--out", default="team_eclectic.csv", help="Output CSV path")
    parser.add_argument("--debug", action="store_true", help="Include debug score breakdown columns")
    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        print(f"❌ File not found: {args.candidates}")
        exit(1)

    run_pipeline(args.candidates, args.out, debug=args.debug)
