from nlp import parse_text
import json

# Read sample data
with open('sample_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Parse the text
records = parse_text(text)

# Display summary
print(f'Total records: {len(records)}')
unknown_count = sum(1 for r in records if r["healer"] == "Unknown")
print(f'Unknown healers: {unknown_count}')
print(f'\nAll healers found: {", ".join(set(r["healer"] for r in records))}')

# Display each record
print('\n' + '='*80)
print('DETAILED RECORDS:')
print('='*80)
for i, r in enumerate(records, 1):
    print(f'\n{i}. Healer: {r["healer"]}')
    print(f'   Cure: {r["cure"] if r["cure"] else "N/A"}')
    print(f'   Symptom: {r["symptom"] if r["symptom"] else "N/A"}')
    print(f'   Outcome: {r["outcome"][:50] if r["outcome"] else "N/A"}...')
    print(f'   Sentiment: {r["sentiment"]}')
    print(f'   Raw: {r["raw"][:70]}...')
