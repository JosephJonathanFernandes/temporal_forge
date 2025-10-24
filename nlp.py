import re
from typing import List, Dict, Tuple

# Lightweight rule-based parser for healer text
# Returns list of dicts: healer, cure, symptom, outcome, sentiment

POSITIVE_KEYWORDS = [
    "worked", "improved", "healed", "helped", "recovered", "good", "successful", "success", "well", "aided", "broke"
]
NEGATIVE_KEYWORDS = [
    "poor", "failed", "didn't", "did not", "no help", "no improvement", "worse", "ineffective", "not help", "bad", "worsened", "no improvement", "nothing changed"
]

def classify_sentiment(text: str) -> str:
    t = (text or '').lower()
    # stronger negative phrases first
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in t)
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in t)
    # heuristics for mixed or weak signals
    if neg_hits > pos_hits and neg_hits > 0:
        return "negative"
    if pos_hits > neg_hits and pos_hits > 0:
        return "positive"
    # handle qualifiers
    if any(p in t for p in ['helped a bit', 'helped slightly', 'some improvement', 'helped a little']):
        return 'positive'
    if any(p in t for p in ["didn't help", "did not help", 'no improvement', 'no help']):
        return 'negative'
    return "neutral"


def extract_healer(text: str) -> str:
    """Extract healer name from text with improved pattern matching."""
    # Pattern 1: Titles + Names (Healer Anna, Dr. Old, Elder Mira, etc.)
    # Handle "Dr." specially to avoid splitting "Dr. Old" incorrectly
    m = re.search(r"(?:Healer|Dr\.?|Doctor|Elder|Brother|Sister|Sr|Mrs|Mr|Ms)\s+([A-Z][a-zA-Z'-.]+)", text)
    if m:
        return m.group(1)
    
    # Pattern 2: Single letter healers (Healer A, Healer B, etc.)
    m2 = re.search(r"Healer\s+([A-Z])\b", text)
    if m2:
        return m2.group(1)
    
    # Pattern 3: Name at sentence start with action verbs
    m3 = re.search(r"^([A-Z][a-zA-Z'-.]+)\s+(used|applied|tried|administered|gave|brewed|attempted|prepared|made|created)", text)
    if m3:
        name = m3.group(1)
        # Filter out common false positives
        if name.lower() not in ['the', 'a', 'an', 'and', 'but', 'or', 'for']:
            return name
    
    # Pattern 4: Fallback - capitalized word before action verbs anywhere in text
    m4 = re.search(r"\b([A-Z][a-zA-Z]{2,})\s+(used|applied|tried|administered|gave|brewed|attempted|prepared)", text)
    if m4:
        name = m4.group(1)
        # Filter out common words
        if name.lower() not in ['healer', 'doctor', 'elder', 'brother', 'sister', 'the', 'a', 'an']:
            return name
    
    return "Unknown"


def extract_cure_and_symptom(text: str) -> Tuple[str, str, str]:
    # Try a few patterns: "used X for Y", "tried X for Y", "used X against Y"
    patterns = [
        r"used\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"tried\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"used\s+([a-zA-Z0-9\s'-]+?)\s+against\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"applied\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"administered\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"gave\s+([a-zA-Z0-9\s'-]+?)\s+to\s+patients?\s+with\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"used\s+a\s+poultice\s+of\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cure = m.group(1).strip()
            symptom = m.group(2).strip()
            outcome = (m.group(3) or "").strip()
            return cure, symptom, outcome

    # fallback: try to find the word after 'used' or 'applied' and optionally a following 'for'
    m2 = re.search(r"(?:used|applied|administered|gave|tried)\s+([a-zA-Z0-9\s'-]+)", text, flags=re.IGNORECASE)
    if m2:
        cure = m2.group(1).split(" for ")[0].strip()
        # try symptom
        m3 = re.search(r"for\s+([a-zA-Z0-9\s'-]+)", text, flags=re.IGNORECASE)
        symptom = m3.group(1).strip() if m3 else ""
        # outcome is rest of sentence after comma or dash
        parts = re.split(r",|-|—|;", text)
        outcome = parts[1].strip() if len(parts) > 1 else ""
        return cure, symptom, outcome

    return "", "", ""


def parse_text(text: str) -> List[Dict]:
    """Parse multi-line unstructured healer text into structured records.

    Each line/sentence ideally contains a record.
    """
    records = []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.replace('\r', ' ')).strip()
    # Split ONLY by newlines and semicolons (keep em-dashes as they connect outcomes)
    raw_lines = []
    for part in re.split(r"\n+|;", text):
        part = part.strip()
        if not part:
            continue
        # DON'T split on periods if they're part of titles (Dr., Mr., etc.)
        # Only split on sentence-ending punctuation followed by capital letter or common starters
        pieces = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z]|Healer|Dr|Doctor|Elder|Brother|Sister)", part) if s.strip()]
        raw_lines.extend(pieces)
    lines = raw_lines

    for line in lines:
        healer = extract_healer(line)
        cure, symptom, outcome = extract_cure_and_symptom(line)
        
        # Skip fragments without healers AND without cure information (likely sentence fragments)
        # Also skip very short fragments that start with lowercase (continuation sentences)
        if healer == "Unknown" and not cure and len(line.split()) < 5:
            continue
        if healer == "Unknown" and not cure and line and line[0].islower():
            continue
        
        # if outcome blank, try to capture trailing sentiment words
        if not outcome:
            # everything after last comma
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 1:
                outcome = parts[-1]
            else:
                outcome = ""

        # if outcome still empty, attempt to extract clause after dash
        if not outcome and '—' in line:
            outcome = line.split('—')[-1].strip()

        # sentiment should consider both outcome and full line for context
        sentiment = classify_sentiment(outcome if outcome else line)

        rec = {
            "healer": healer,
            "cure": cure or "",
            "symptom": symptom or "",
            "outcome": outcome or "",
            "sentiment": sentiment,
            "raw": line,
        }
        records.append(rec)

    return records


if __name__ == "__main__":
    s = "Healer A used herb willow for fever, it worked well.\nHealer B used honey for cough, patients improved.\nHealer C tried willow for infection but results were poor."
    print(parse_text(s))
