import os

# 1. Update rank.py
with open("rank.py", "r", encoding="utf-8") as f:
    rank_content = f.read()

rank_content = rank_content.replace("disqualifier_mult = check_disqualifier(candidate)", "disqualifier_mult = score_nlp_context(candidate)")

with open("rank.py", "w", encoding="utf-8") as f:
    f.write(rank_content)


# 2. Update generate_notebook.py
with open("generate_notebook.py", "r", encoding="utf-8") as f:
    nb_content = f.read()

nb_content = nb_content.replace('"def check_disqualifier")', '"def score_nlp_context")')

with open("generate_notebook.py", "w", encoding="utf-8") as f:
    f.write(nb_content)

print("Fixed variables and markers")
