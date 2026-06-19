import json
import collections
import re

ngram_counts = collections.Counter()

with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip(): continue
        
        c = json.loads(line)
        profile = c.get('profile', {})
        career = c.get('career_history', [])
        
        summary = profile.get('summary', '').lower()
        descriptions = [j.get('description', '').lower() for j in career]
        
        all_text = summary + " " + " ".join(descriptions)
        
        # Skip candidates with known negative caveats
        if any(bad in all_text for bad in ['chatgpt', 'consulting', 'transition', 'exposure', 'curious']):
            continue
            
        sentences = [s.strip() for s in all_text.split('.') if s.strip()]
        for s in sentences:
            # looking for specific models or hardcore ML words
            if 'model' in s or 'pipeline' in s or 'production' in s or 'retrieval' in s or 'system' in s:
                words = re.findall(r'\b[a-z]{3,}\b', s)
                for j in range(len(words)-2):
                    ngram = " ".join(words[j:j+3])
                    ngram_counts[ngram] += 1
                    
        if i >= 10000:
            break

print("Top positive ML-related 3-grams:")
for k, v in ngram_counts.most_common(50):
    if v > 100:
        print(f"  {k}: {v}")
