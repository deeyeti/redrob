import json
import collections
import re
import pandas as pd

df = pd.read_csv('team_eclectic.csv')
top_ids = set(df['candidate_id'].tolist())

top_ngrams = collections.Counter()

with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        if c.get('candidate_id') in top_ids:
            profile = c.get('profile', {})
            career = c.get('career_history', [])
            
            summary = profile.get('summary', '').lower()
            descriptions = [j.get('description', '').lower() for j in career]
            all_text = summary + " " + " ".join(descriptions)
            
            sentences = [s.strip() for s in all_text.split('.') if s.strip()]
            for s in sentences:
                words = re.findall(r'\b[a-z]{3,}\b', s)
                for j in range(len(words)-2):
                    ngram = " ".join(words[j:j+3])
                    top_ngrams[ngram] += 1
            
            if len(top_ngrams) > 1000000: break

print("Top n-grams in the actual Top 100 candidates:")
for k, v in top_ngrams.most_common(50):
    if v > 10:
        print(f"  {k}: {v}")
