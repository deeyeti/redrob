import os

with open("rank.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add imports (re, collections, concurrent.futures)
if "import re" not in content:
    content = content.replace("import pandas as pd", "import pandas as pd\nimport re\nimport collections\nimport concurrent.futures")

# 2. Add SKILL_SYNONYMS and JD_TEXT after JD_NICE_SKILLS
synonyms_block = """
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

JD_TEXT = \"\"\"
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
\"\"\"

# Basic pre-computed TF-IDF for JD
def tokenize(text):
    return re.findall(r'\\b[a-z0-9]+\\b', text.lower())

JD_TOKENS = tokenize(JD_TEXT)
JD_TERM_FREQ = collections.Counter(JD_TOKENS)
JD_VOCAB_SIZE = len(JD_TERM_FREQ)
"""
if "SKILL_SYNONYMS =" not in content:
    content = content.replace("# ============================================================\n# TITLE / ROLE TAXONOMY", synonyms_block + "\n# ============================================================\n# TITLE / ROLE TAXONOMY")

# 3. Update score_skills with normalization and TF-IDF
score_skills_old = """    # Normalize skill names from candidate profile
    skill_map = {}  # name_lower -> {proficiency, duration_months, endorsements}
    for sk in skills:
        name_lower = sk.get("name", "").lower()
        skill_map[name_lower] = sk"""

score_skills_new = """    # Normalize skill names from candidate profile
    skill_map = {}  # name_lower -> {proficiency, duration_months, endorsements}
    for sk in skills:
        name_lower = sk.get("name", "").lower()
        name_lower = SKILL_SYNONYMS.get(name_lower, name_lower)
        skill_map[name_lower] = sk"""
content = content.replace(score_skills_old, score_skills_new)

tfidf_old = """    # Normalize separately then combine
    core_norm = core_score / core_max if core_max > 0 else 0.0
    nice_norm = nice_score / nice_max if nice_max > 0 else 0.0

    final = 0.80 * core_norm + 0.20 * nice_norm"""

tfidf_new = """    # Normalize separately then combine
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

    final = 0.60 * core_norm + 0.15 * nice_norm + 0.25 * tfidf_norm"""
content = content.replace(tfidf_old, tfidf_new)

# 4. Quantified impact detection
impact_old = """    # 2. Count production/impact signals
    found_prod = sum(1 for sig in PRODUCTION_SIGNALS if sig in all_text)
    # Normalize: 6+ production signals = full score
    prod_score = min(found_prod / 6.0, 1.0)"""

impact_new = """    # 2. Count production/impact signals
    found_prod = sum(1 for sig in PRODUCTION_SIGNALS if sig in all_text)
    
    # 2b. Quantified impact detection (e.g. 50%, 10x, 20ms, 5 million)
    quant_patterns = [
        r'\\b\\d+\\s*%', r'\\b\\d+\\s*x\\b', r'\\b\\d+\\s*ms\\b', 
        r'\\b\\d+\\s*(?:million|billion|k|m|b)\\b', r'\\$\\d+'
    ]
    found_quant = sum(1 for p in quant_patterns if re.search(p, all_text))
    
    # Normalize: 6+ production signals + 2+ quant signals = full score
    prod_score = min((found_prod + found_quant) / 8.0, 1.0)"""
content = content.replace(impact_old, impact_new)

# 5. Notice period fix
notice_old = """    # Notice period scoring
    if notice_days <= 15:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.85"""

notice_new = """    # Notice period scoring
    if notice_days <= 30:
        notice_score = 1.0"""
content = content.replace(notice_old, notice_new)

# 6. Parallel processing in run_pipeline
pipeline_old = """def run_pipeline(candidates_path: str, output_path: str, debug: bool = False):
    results = []
    total = 0
    honeypot_count = 0
    disqualified_count = 0

    for candidate in load_candidates(candidates_path):
        total += 1
        result = score_candidate(candidate)
        if result is None:
            honeypot_count += 1
            continue
        if result["_disqualifier"] < 0.5:
            disqualified_count += 1
        results.append(result)

        if total % 10000 == 0:
            print(f"  Processed {total:,} candidates... ({len(results):,} scored)")"""

pipeline_new = """def run_pipeline(candidates_path: str, output_path: str, debug: bool = False):
    results = []
    total = 0
    honeypot_count = 0
    disqualified_count = 0

    candidates_list = list(load_candidates(candidates_path))
    total = len(candidates_list)
    print(f"Loaded {total:,} candidates. Scoring in parallel...")

    # Parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(score_candidate, candidates_list, chunksize=2000):
            if result is None:
                honeypot_count += 1
                continue
            if result["_disqualifier"] < 0.5:
                disqualified_count += 1
            results.append(result)"""
content = content.replace(pipeline_old, pipeline_new)

with open("rank.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated rank.py successfully")
