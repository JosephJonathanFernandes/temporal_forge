import re
from typing import List, Dict

# Lightweight rule-based parser for healer text
# Returns list of dicts: healer, cure, symptom, outcome, sentiment

POSITIVE_KEYWORDS = [
    "worked", "improved", "healed", "helped", "recovered", "good", "successful", "success", "well"
]
NEGATIVE_KEYWORDS = [
    "poor", "failed", "didn't", "did not", "no help", "no improvement", "worse", "ineffective", "not help", "bad"
]

def classify_sentiment(text: str) -> str:
    t = text.lower()
    for kw in POSITIVE_KEYWORDS:
        if kw in t:
            return "positive"
    for kw in NEGATIVE_KEYWORDS:
        if kw in t:
            return "negative"
    # neutral by default
    return "neutral"


def extract_healer(text: str) -> str:
    # match "Healer X" or "Healer: X" or names like "Healer Anna"
    m = re.search(r"Healer\s+([A-Z][a-zA-Z'-]+)", text)
    if m:
        return m.group(1)
    # try generic name patterns at sentence start
    m2 = re.search(r"^([A-Z][a-zA-Z'-]+)\s+used", text)
    if m2:
        return m2.group(1)
    return "Unknown"


def extract_cure_and_symptom(text: str) -> (str, str, str):
    # Try a few patterns: "used X for Y", "tried X for Y", "used X against Y"
    patterns = [
        r"used\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"tried\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"used\s+([a-zA-Z0-9\s'-]+?)\s+against\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
        r"applied\s+([a-zA-Z0-9\s'-]+?)\s+for\s+([a-zA-Z0-9\s'-]+)[,\.-]?(.*)$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cure = m.group(1).strip()
            symptom = m.group(2).strip()
            outcome = (m.group(3) or "").strip()
            return cure, symptom, outcome

    # fallback: try to find the word after 'used' and optionally a following 'for'
    m2 = re.search(r"used\s+([a-zA-Z0-9\s'-]+)", text, flags=re.IGNORECASE)
    if m2:
        cure = m2.group(1).split(" for ")[0].strip()
        # try symptom
        m3 = re.search(r"for\s+([a-zA-Z0-9\s'-]+)", text, flags=re.IGNORECASE)
        symptom = m3.group(1).strip() if m3 else ""
        # outcome is rest of sentence after comma
        parts = re.split(r",|-|—", text)
        outcome = parts[1].strip() if len(parts) > 1 else ""
        return cure, symptom, outcome

    return "", "", ""


def parse_text(text: str) -> List[Dict]:
    """Parse multi-line unstructured healer text into structured records.

    Each line/sentence ideally contains a record.
    """
    records = []
    # split into lines, but also into sentences as fallback
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        # split into sentences by period
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    for line in lines:
        healer = extract_healer(line)
        cure, symptom, outcome = extract_cure_and_symptom(line)
        # if outcome blank, try to capture trailing sentiment words
        if not outcome:
            # everything after last comma
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 1:
                outcome = parts[-1]
            else:
                outcome = ""

        sentiment = classify_sentiment(outcome)

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
