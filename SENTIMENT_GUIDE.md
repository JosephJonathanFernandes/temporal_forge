# 🎭 Sentiment Classification - How It Works

## 📊 The Algorithm

```
INPUT: "it worked well"
         ↓
    LOWERCASE: "it worked well"
         ↓
    COUNT KEYWORDS:
    ├─ POSITIVE: "worked" ✓, "well" ✓  → pos_hits = 2
    └─ NEGATIVE: (none)                → neg_hits = 0
         ↓
    COMPARE:
    ├─ neg_hits (0) > pos_hits (2)? NO
    └─ pos_hits (2) > neg_hits (0)? YES ✓
         ↓
    RESULT: POSITIVE ✅
```

---

## 🎯 Decision Tree

```
                    START
                      ↓
           Convert text to lowercase
                      ↓
        Count positive & negative keywords
                      ↓
            ┌─────────┴─────────┐
            │                   │
    neg_hits > pos_hits?   pos_hits > neg_hits?
         YES ↓                  ↓ YES
            │                   │
    Are neg_hits > 0?    Are pos_hits > 0?
         YES ↓                  ↓ YES
            │                   │
        NEGATIVE ❌          POSITIVE ✅
            │                   │
            └─────────┬─────────┘
                      ↓ (Neither condition met)
              Check special phrases:
              ├─ "helped a bit" → POSITIVE
              ├─ "some improvement" → POSITIVE  
              ├─ "didn't help" → NEGATIVE
              └─ "no improvement" → NEGATIVE
                      ↓ (No matches)
                  NEUTRAL ○
```

---

## 📝 Examples by Category

### ✅ POSITIVE Outcomes
| Text | Matched Keywords | Result |
|------|-----------------|--------|
| "it worked well" | worked, well | POSITIVE |
| "patients healed quickly" | healed | POSITIVE |
| "fever broke" | broke | POSITIVE |
| "helped a bit" | helped (+ special phrase) | POSITIVE |
| "some improvement noted" | (special phrase) | POSITIVE |

**Why:** More positive keywords than negative, or special positive phrases detected

---

### ❌ NEGATIVE Outcomes
| Text | Matched Keywords | Result |
|------|-----------------|--------|
| "results were poor" | poor | NEGATIVE |
| "it didn't help" | didn't (+ special phrase) | NEGATIVE |
| "condition worsened" | worse, worsened | NEGATIVE |
| "no improvement observed" | no improvement (x2) | NEGATIVE |

**Why:** More negative keywords than positive, or special negative phrases detected

---

### ○ NEUTRAL Outcomes
| Text | Matched Keywords | Result |
|------|-----------------|--------|
| "used honey for cough" | (none) | NEUTRAL |
| "applied crushed mint" | (none) | NEUTRAL |
| "worked but results were poor" | worked + poor (tied 1:1) | NEUTRAL |

**Why:** No keywords found, or equal positive/negative keywords (tie)

---

## 🔧 The Code

```python
POSITIVE_KEYWORDS = [
    "worked", "improved", "healed", "helped", "recovered", 
    "good", "successful", "success", "well", "aided", "broke"
]

NEGATIVE_KEYWORDS = [
    "poor", "failed", "didn't", "did not", "no help", 
    "no improvement", "worse", "ineffective", "not help", 
    "bad", "worsened", "nothing changed"
]

def classify_sentiment(text: str) -> str:
    t = (text or '').lower()
    
    # Count keyword matches
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in t)
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in t)
    
    # Compare counts
    if neg_hits > pos_hits and neg_hits > 0:
        return "negative"
    if pos_hits > neg_hits and pos_hits > 0:
        return "positive"
    
    # Handle special phrases
    if any(p in t for p in ['helped a bit', 'helped slightly', 
                             'some improvement', 'helped a little']):
        return 'positive'
    if any(p in t for p in ["didn't help", "did not help", 
                             'no improvement', 'no help']):
        return 'negative'
    
    return "neutral"
```

---

## 💡 Key Points

1. **Simple & Fast**: Rule-based, no ML models needed
2. **Keyword Counting**: Counts how many positive vs negative words appear
3. **Special Phrases**: Handles common medical phrases that might be missed
4. **Tie Breaking**: Equal counts = NEUTRAL
5. **Case Insensitive**: Converts to lowercase before matching

---

## 🎯 Edge Cases

### Tied Scores (1 positive + 1 negative)
```
"worked but results were poor"
├─ pos_hits = 1 (worked)
├─ neg_hits = 1 (poor)
└─ Result: NEUTRAL (tie)
```

### Multiple Matches
```
"no improvement observed"
├─ Matches "no improvement" TWICE (substring overlap)
├─ neg_hits = 2
└─ Result: NEGATIVE
```

### No Keywords
```
"used honey for cough"
├─ No matches in either list
└─ Result: NEUTRAL (default)
```

---

## 🚀 How to Improve

1. **Add More Keywords**
   ```python
   POSITIVE_KEYWORDS += ["effective", "cured", "better", "relief"]
   NEGATIVE_KEYWORDS += ["useless", "harmful", "deteriorated"]
   ```

2. **Use VADER Sentiment Analysis**
   ```python
   from nltk.sentiment import SentimentIntensityAnalyzer
   sia = SentimentIntensityAnalyzer()
   score = sia.polarity_scores(text)
   # Returns: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.5}
   ```

3. **Use Transformers (Advanced)**
   ```python
   from transformers import pipeline
   classifier = pipeline("sentiment-analysis")
   result = classifier(text)
   # Returns: [{'label': 'POSITIVE', 'score': 0.999}]
   ```

---

## 📊 Current Performance

From your sample data (11 records):
- ✅ **4 Positive**: A (worked well), Anna (healed quickly), Mira (helped a bit), Signe (broke)
- ❌ **3 Negative**: C (poor), John (didn't help), Liao (worsened), Noor (no improvement)
- ○ **4 Neutral**: B, Old, Tomas (no outcome keywords)

**Accuracy**: Good for clear cases, but may miss nuanced medical language.
