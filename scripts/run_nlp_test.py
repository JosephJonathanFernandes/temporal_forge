import sys
from pathlib import Path
# ensure project root is on sys.path when running from scripts/
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from models.nlp_pipeline import process_scrolls, extract_entities, classify_record, topics_from_texts
s = open('sample_input.txt','r',encoding='utf-8').read()
res = process_scrolls(s)
print('records:', len(res.get('records',[])))
print('pos_counts:', res.get('cures_pos_counts'))
print('neg_counts:', res.get('cures_neg_counts'))
print('keywords:', res.get('keywords')[:10])
ents = extract_entities(s)
print('entities sample:', ents)
texts = [r.get('raw','') for r in res.get('records',[])]
print('topics:', topics_from_texts(texts, top_n=8))
# show labels for first few
for r in res.get('records',[])[:8]:
    print('raw->', r.get('raw'))
    print('  cure,symptom,sentiment,label ->', r.get('cure'), r.get('symptom'), r.get('sentiment'), classify_record(r))
