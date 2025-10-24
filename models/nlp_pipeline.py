"""NLP pipeline wrapper for The Healer's Scribe

This module provides `process_scrolls(text)` which returns a serializable
result dict containing extracted cures, symptoms, keyword list, summary and
counts. It will use heavier libraries (spaCy, nltk VADER, sklearn, transformers)
when installed, but falls back to the lightweight `nlp.parse_text` heuristics.
"""
from typing import List, Dict, Any
import logging
import re

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

import pandas as pd
from nlp import parse_text

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace('\r', ' ')).strip()


def extract_keywords_tfidf(texts: List[str], top_n: int = 10) -> List[str]:
    if not SKLEARN_AVAILABLE:
        # fallback to simple frequency
        words = ' '.join(texts).lower().split()
        freq = {}
        for w in words:
            if len(w) < 3:
                continue
            freq[w] = freq.get(w, 0) + 1
        return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])][:top_n]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1
    indices = scores.argsort()[-top_n:][::-1]
    return [vectorizer.get_feature_names_out()[i] for i in indices]


def summarize_with_transformer(text: str) -> str:
    if not TRANSFORMERS_AVAILABLE:
        return ""  # caller will handle fallback
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        res = summarizer(text, max_length=120, min_length=20, do_sample=False)
        return res[0]['summary_text']
    except Exception as e:
        logger.warning("Transformer summarization failed: %s", e)
        return ""


def analyze_sentiments_vader(outcomes: List[str]) -> List[float]:
    if not VADER_AVAILABLE:
        # fallback: map 'positive'/'negative' strings if present
        scores = []
        for o in outcomes:
            s = o.lower()
            if any(k in s for k in ['work', 'improv', 'heal', 'help']):
                scores.append(0.6)
            elif any(k in s for k in ['poor', 'fail', "didn't", 'did not', 'no help', 'worse']):
                scores.append(-0.6)
            else:
                scores.append(0.0)
        return scores

    sid = SentimentIntensityAnalyzer()
    return [sid.polarity_scores(o)['compound'] for o in outcomes]


def process_scrolls(text: str) -> Dict[str, Any]:
    """Process raw healer scrolls and return structured insights.

    Returns a dict with keys:
      - records: list of parsed records (healer,cure,symptom,outcome,sentiment,raw)
      - cures_pos_counts: dict cure -> positive count
      - cures_neg_counts: dict cure -> negative count
      - keywords: list of top keywords
      - summary: text summary (transformer if available, else rule-based)
    """
    text = clean_text(text)
    records = parse_text(text)
    df = pd.DataFrame(records)

    # compute counts
    cures_pos = {}
    cures_neg = {}
    if not df.empty:
        for _, r in df.iterrows():
            cure = (r.get('cure') or '').strip()
            s = r.get('sentiment')
            if not cure:
                continue
            if s == 'positive':
                cures_pos[cure] = cures_pos.get(cure, 0) + 1
            elif s == 'negative':
                cures_neg[cure] = cures_neg.get(cure, 0) + 1

    # keywords from cures/symptoms/raw
    texts_for_k = []
    if not df.empty:
        texts_for_k = (df['cure'].fillna('') + ' ' + df['symptom'].fillna('') + ' ' + df['raw'].fillna('')).tolist()
    else:
        texts_for_k = [text]

    keywords = extract_keywords_tfidf(texts_for_k, top_n=12)

    # sentiment scores for outcomes
    outcomes = df['outcome'].fillna('').tolist() if not df.empty else [text]
    sentiment_scores = analyze_sentiments_vader(outcomes)

    # try transformer summarization
    summary = ''
    if TRANSFORMERS_AVAILABLE:
        summary = summarize_with_transformer(text)

    if not summary:
        # rule-based summary: list top positive and negative cures
        pos_sorted = [k for k, _ in sorted(cures_pos.items(), key=lambda x: -x[1])]
        neg_sorted = [k for k, _ in sorted(cures_neg.items(), key=lambda x: -x[1])]
        parts = []
        if pos_sorted:
            parts.append(f"Frequent effective cures: {', '.join(pos_sorted[:5])}.")
        if neg_sorted:
            parts.append(f"Frequent ineffective cures: {', '.join(neg_sorted[:5])}.")
        if not parts:
            parts = ["No clear wisdom extracted — add more notes or enable transformer summarization."]
        summary = ' '.join(parts)

    result = {
        'records': records,
        'cures_pos_counts': cures_pos,
        'cures_neg_counts': cures_neg,
        'keywords': keywords,
        'summary': summary,
        'sentiment_scores': sentiment_scores,
    }

    return result


if __name__ == '__main__':
    sample = "Healer Anna used garlic for infections — patients healed quickly. Healer John used saltwater for fever — it didn't help."
    print(process_scrolls(sample))
