"""
Generates eclectic_pipeline.ipynb — a fully self-contained notebook
that embeds all the ranking code directly (no imports from rank.py).
"""
import json

def cell(cell_type, source, **kwargs):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
        **kwargs
    }

# Read rank.py to get the full scoring code
with open("rank.py", "r", encoding="utf-8") as f:
    rank_src = f.read()

# Helper to extract a block from rank.py by start/end sentinel strings
def extract(src, start_marker, end_marker=None):
    s = src.find(start_marker)
    if s == -1:
        raise ValueError(f"Marker not found: {start_marker!r}")
    if end_marker:
        e = src.find(end_marker, s)
        if e == -1:
            raise ValueError(f"End marker not found: {end_marker!r}")
        return src[s:e].rstrip()
    # Go to next top-level def/class or end of file
    lines = src[s:].split("\n")
    result = []
    for i, line in enumerate(lines):
        if i > 0 and line and not line[0].isspace() and not line.startswith("#") and line.strip():
            break
        result.append(line)
    return "\n".join(result).rstrip()

# ── Extract sections ──────────────────────────────────────────────────────────
imports_block = """\
import os
import json
import argparse
import math
from datetime import date, datetime
from typing import Optional, Tuple, List
import pandas as pd
import re
import collections
import concurrent.futures

TODAY = date.today()
print(f"Pipeline ready. Today = {TODAY}")
"""

taxonomy_block = extract(rank_src, "JD_CORE_SKILLS", "# ============================================================\n# TITLE")
taxonomy_block += "\n\n"
taxonomy_block += extract(rank_src, "TITLE_TIERS", "TITLE_SCORE_MAP")
taxonomy_block += "\n\n"
taxonomy_block += extract(rank_src, "TITLE_SCORE_MAP", "# Consulting")
taxonomy_block += "\n\n"
taxonomy_block += extract(rank_src, "# Consulting firms", "# Industries")
taxonomy_block += "\n\n"
taxonomy_block += extract(rank_src, "# Industries that signal", "# Production")
taxonomy_block += "\n\n"
taxonomy_block += extract(rank_src, "# Production/impact signals", "# Keywords in career")
taxonomy_block += "\n\n"
taxonomy_block += extract(rank_src, "# Keywords in career descriptions", "TODAY")
taxonomy_block = taxonomy_block.rstrip()

helpers_block = extract(rank_src, "def days_since", "# ============================================================\n# SCORING COMPONENTS")


scoring_block = "\n\n".join([
    extract(rank_src, "def score_skills", "def score_title"),
    extract(rank_src, "def score_title", "def score_yoe"),
    extract(rank_src, "def score_yoe", "def score_behavioral"),
    extract(rank_src, "def score_behavioral", "def score_education"),
    extract(rank_src, "def score_education", "def score_career_depth"),
    extract(rank_src, "def score_career_depth", "def score_company_context"),
    extract(rank_src, "def score_company_context", "def score_availability"),
    extract(rank_src, "def score_availability", "def score_nlp_context"),
    extract(rank_src, "def score_nlp_context", "# ============================================================\n# REASONING"),
])

reasoning_block = extract(rank_src, "def generate_reasoning", "# ============================================================\n# MAIN")

score_candidate_block = extract(rank_src, "def score_candidate", "# ============================================================\n# PIPELINE")

pipeline_block = "\n\n".join([
    extract(rank_src, "def open_file", "def load_candidates"),
    extract(rank_src, "def load_candidates", "def run_pipeline"),
    extract(rank_src, "def run_pipeline", "# ============================================================\n# ENTRY"),
])

run_block = """\
# ── Configuration ─────────────────────────────────────────────────────────────
import os, time

if os.path.exists("candidates.jsonl"):
    DATA_PATH = "candidates.jsonl"
elif os.path.exists("candidates.jsonl.gz"):
    DATA_PATH = "candidates.jsonl.gz"
elif os.path.exists("sample_candidates.json"):
    DATA_PATH = "sample_candidates.json"
    print("⚠️  Using sample_candidates.json (50 candidates only)")
else:
    raise FileNotFoundError("❌ No candidates file found — place candidates.jsonl here")

OUTPUT_PATH = "team_eclectic.csv"

print(f"📂 Input  : {DATA_PATH}")
print(f"📝 Output : {OUTPUT_PATH}")
print()

start = time.time()
df_result = run_pipeline(DATA_PATH, OUTPUT_PATH)
elapsed = time.time() - start
print(f"\\n⏱️  Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
"""

validate_block = """\
# Run official format validator
import subprocess, sys
result = subprocess.run([sys.executable, "validate_submission.py", "team_eclectic.csv"],
                       capture_output=True, text=True)
print(result.stdout or result.stderr)
"""

inspect_block = """\
import pandas as pd

df = pd.read_csv("team_eclectic.csv")
print(f"Total rows   : {len(df)}")
print(f"Score range  : {df['score'].min():.4f} – {df['score'].max():.4f}")
print(f"Score std    : {df['score'].std():.4f}")
print()
print("═" * 110)
print(f"{'#':>4}  {'Candidate':15}  {'Score':>6}  Reasoning")
print("═" * 110)
for _, r in df.head(20).iterrows():
    print(f"  #{int(r['rank']):3d}  {r['candidate_id']:15}  {r['score']:.4f}  {r['reasoning'][:80]}")
"""

# ── Build notebook cells ──────────────────────────────────────────────────────
cells = [
    cell("markdown", [
        "# 🌌 Team Eclectic — Intelligent Candidate Ranker\n",
        "\n",
        "> **Redrob India Runs Data & AI Challenge** — Senior AI Engineer Role  \n",
        "> Fully self-contained notebook. No external imports needed beyond `pandas`.\n",
        "\n",
        "## Scoring Architecture\n",
        "\n",
        "| Component | Weight | Signal |\n",
        "|---|---|---|\n",
        "| Skill Match | 35% | JD taxonomy (60+ keywords) × proficiency × duration |\n",
        "| Title/Role | 17% | ML/AI Engineer → SWE → Non-tech (4-tier) |\n",
        "| YoE Fit | 13% | Sweet spot 6–8 yrs; JD range 5–9 |\n",
        "| Behavioral | 14% | Response rate, interview rate, GitHub, recency, saved |\n",
        "| Career Depth | 7% | ML keyword breadth + production deployment signals in descriptions |\n",
        "| Company Context | 6% | Product/tech co ratio vs consulting; industry bonus |\n",
        "| Education | 4% | Institution tier + field relevance |\n",
        "| Availability | 4% | Notice period + open-to-work + location |\n",
        "\n",
        "## Honeypot Detection (11 checks)\n",
        "Career date reversal · Fake expert skills (zero months) · Impossible YoE · Education date reversal · "
        "Inverted salary range · Triple-perfect behavioral signals · YoE vs career duration mismatch · "
        "All assessment scores = 100 · Skill duration > career length · Duplicate jobs · "
        "is_current=True with past end_date\n",
        "\n",
        "**Runtime**: ~50s for 100K candidates · CPU only · No network · ≤2GB RAM",
    ]),

    cell("markdown", ["## 1. Imports"]),
    cell("code", imports_block),

    cell("markdown", ["## 2. JD Skill Taxonomy & Role Tiers"]),
    cell("code", taxonomy_block),

    cell("markdown", ["## 3. Helper Functions"]),
    cell("code", helpers_block),

    cell("markdown", ["## 4. Scoring Components"]),
    cell("code", scoring_block),

    cell("markdown", ["## 6. Reasoning Generation"]),
    cell("code", reasoning_block),

    cell("markdown", ["## 7. Main Scoring Function"]),
    cell("code", score_candidate_block),

    cell("markdown", ["## 8. Pipeline (Load → Score → Rank → Output)"]),
    cell("code", pipeline_block),

    cell("markdown", [
        "## 9. Run Pipeline\n",
        "\n",
        "Place `candidates.jsonl` in the same directory, then run this cell.\n",
        "The `sample_candidates.json` (50 candidates) is used automatically if `candidates.jsonl` is not found.",
    ]),
    cell("code", run_block),

    cell("markdown", ["## 10. Validate Submission Format"]),
    cell("code", validate_block),

    cell("markdown", ["## 11. Inspect Top Results"]),
    cell("code", inspect_block),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("eclectic_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ eclectic_pipeline.ipynb generated successfully")
print(f"   Total cells: {len(cells)}")
