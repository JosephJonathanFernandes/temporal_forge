from nlp import classify_sentiment, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

print("=" * 80)
print("🎭 SENTIMENT CLASSIFICATION SYSTEM")
print("=" * 80)

print("\n📊 KEYWORD LISTS:")
print(f"\n✅ POSITIVE KEYWORDS ({len(POSITIVE_KEYWORDS)}):")
print(f"   {', '.join(POSITIVE_KEYWORDS)}")
print(f"\n❌ NEGATIVE KEYWORDS ({len(NEGATIVE_KEYWORDS)}):")
print(f"   {', '.join(NEGATIVE_KEYWORDS)}")

print("\n" + "=" * 80)
print("🧪 HOW IT WORKS:")
print("=" * 80)
print("""
1. Convert text to lowercase
2. Count positive keyword matches → pos_hits
3. Count negative keyword matches → neg_hits
4. Compare counts:
   - If neg_hits > pos_hits AND neg_hits > 0 → NEGATIVE
   - If pos_hits > neg_hits AND pos_hits > 0 → POSITIVE
   - Otherwise → Check special phrases
5. Special phrases:
   - "helped a bit", "some improvement" → POSITIVE
   - "didn't help", "no improvement" → NEGATIVE
6. If no matches → NEUTRAL
""")

print("=" * 80)
print("🧪 TEST EXAMPLES:")
print("=" * 80)

test_cases = [
    # Clear positive
    ("it worked well", "Should be POSITIVE (has 'worked' and 'well')"),
    ("patients healed quickly", "Should be POSITIVE (has 'healed')"),
    ("patients improved after two days", "Should be POSITIVE (has 'improved')"),
    ("it helped a bit", "Should be POSITIVE (special phrase)"),
    ("some improvement noted", "Should be POSITIVE (special phrase)"),
    ("fever broke", "Should be POSITIVE (has 'broke')"),
    
    # Clear negative
    ("results were poor", "Should be NEGATIVE (has 'poor')"),
    ("it didn't help", "Should be NEGATIVE (special phrase)"),
    ("no improvement observed", "Should be NEGATIVE (special phrase)"),
    ("condition worsened", "Should be NEGATIVE (has 'worsened')"),
    
    # Neutral
    ("used honey for cough", "Should be NEUTRAL (no keywords)"),
    ("applied crushed mint", "Should be NEUTRAL (no keywords)"),
    ("administered willow bark", "Should be NEUTRAL (no keywords)"),
    
    # Mixed/Edge cases
    ("worked but results were poor", "NEGATIVE (neg_hits=1, pos_hits=1, but 'poor' weighs)"),
    ("improved slightly but still bad", "Could go either way - let's see!"),
]

for text, explanation in test_cases:
    result = classify_sentiment(text)
    
    # Show which keywords matched
    text_lower = text.lower()
    pos_matches = [kw for kw in POSITIVE_KEYWORDS if kw in text_lower]
    neg_matches = [kw for kw in NEGATIVE_KEYWORDS if kw in text_lower]
    
    icon = "✓" if result == "positive" else "✗" if result == "negative" else "○"
    color = "\033[92m" if result == "positive" else "\033[91m" if result == "negative" else "\033[93m"
    reset = "\033[0m"
    
    print(f"\n{icon} Text: \"{text}\"")
    print(f"   Result: {color}{result.upper()}{reset}")
    print(f"   Explanation: {explanation}")
    if pos_matches:
        print(f"   ✅ Positive matches: {', '.join(pos_matches)}")
    if neg_matches:
        print(f"   ❌ Negative matches: {', '.join(neg_matches)}")
    if not pos_matches and not neg_matches:
        print(f"   ○ No keyword matches")

print("\n" + "=" * 80)
print("📋 SUMMARY:")
print("=" * 80)
print("""
The sentiment classifier is KEYWORD-BASED and uses simple counting:
- It's fast and doesn't require ML models
- Works well for clear outcomes ("worked well" vs "didn't help")
- Can be fooled by complex sentences with mixed sentiments
- Special phrases handle common medical documentation patterns

To improve accuracy:
1. Add more keywords to POSITIVE_KEYWORDS/NEGATIVE_KEYWORDS
2. Add more special phrases for medical context
3. Consider upgrading to VADER (sentiment analysis library) for better accuracy
4. Or use transformer models (BERT) for state-of-the-art results
""")
