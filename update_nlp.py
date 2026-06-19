import re

# 1. Update rank.py
with open("rank.py", "r", encoding="utf-8") as f:
    rank_content = f.read()

positive_patterns = """
POSITIVE_NLP_PATTERNS = [
    r"fine-tune[d]?.*model",
    r"implemented.*hybrid search",
    r"custom cross-encoder",
    r"deployed learning to rank",
    r"trained.*embeddings",
    r"improved ndcg",
    r"improved mrr",
    r"built.*retrieval system"
]
"""

# Inject POSITIVE_NLP_PATTERNS after CAVEAT_PATTERNS
caveat_end = rank_content.find("]\n\n# Keywords in career descriptions")
if caveat_end != -1:
    rank_content = rank_content[:caveat_end+1] + "\n" + positive_patterns + rank_content[caveat_end+1:]

# Rename check_disqualifier to score_nlp_context
rank_content = rank_content.replace("def check_disqualifier(candidate: dict) -> float:", "def score_nlp_context(candidate: dict) -> float:")
rank_content = rank_content.replace("mult_disqualifier = check_disqualifier(candidate)", "mult_disqualifier = score_nlp_context(candidate)")

# Add positive NLP logic
target_return = "    return multiplier"
positive_logic = """    for pat in POSITIVE_NLP_PATTERNS:
        if __import__("re").search(pat, text_blob):
            multiplier *= 1.2
            break

    return multiplier"""
rank_content = rank_content.replace(target_return, positive_logic, 1)

with open("rank.py", "w", encoding="utf-8") as f:
    f.write(rank_content)


# 2. Update generate_notebook.py
with open("generate_notebook.py", "r", encoding="utf-8") as f:
    nb_content = f.read()

nb_content = nb_content.replace('extract(rank_src, "def check_disqualifier",', 'extract(rank_src, "def score_nlp_context",')

with open("generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(nb_content)

print("Updated rank.py and generate_notebook.py with positive NLP scoring")
