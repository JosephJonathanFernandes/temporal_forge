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
else:
    # try to load small english model if available
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            # sometimes models are available under package name
            _nlp = spacy.load("en")
        except Exception:
            _nlp = None


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


def extract_keywords_spacy(texts: List[str], top_n: int = 10) -> List[str]:
    """Extract candidate keywords using spaCy noun chunks & entities."""
    if not SPACY_AVAILABLE or _nlp is None:
        # fallback to simple frequency
        words = ' '.join(texts).lower().split()
        freq = {}
        for w in words:
            if len(w) < 3:
                continue
            w = re.sub(r"[^a-z0-9'-]", '', w)
            if not w:
                continue
            freq[w] = freq.get(w, 0) + 1
        return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])][:top_n]

    doc = _nlp(' '.join(texts))
    freq = {}
    # nouns, noun_chunks, and entity text
    for chunk in doc.noun_chunks:
        key = chunk.lemma_.lower().strip()
        if len(key) < 3:
            continue
        freq[key] = freq.get(key, 0) + 1
    for ent in doc.ents:
        key = ent.lemma_.lower().strip()
        if len(key) < 3:
            continue
        freq[key] = freq.get(key, 0) + 2

    # return top_n
    return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])][:top_n]


def extract_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    """Select keyword extraction strategy based on available libs."""
    if SKLEARN_AVAILABLE:
        return extract_keywords_tfidf(texts, top_n=top_n)
    if SPACY_AVAILABLE:
        return extract_keywords_spacy(texts, top_n=top_n)
    # fallback
    return extract_keywords_tfidf(texts, top_n=top_n)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract simple entity lists: healers, treatments, symptoms, diseases.

    Uses spaCy entities and noun chunks when available; falls back to simple pattern
    and frequency-based extraction.
    """
    treatments = []
    symptoms = []
    healers = []
    diseases = []

    if SPACY_AVAILABLE and _nlp is not None:
        try:
            doc = _nlp(text)
            # Healers: PERSON or titles
            for ent in doc.ents:
                if ent.label_ in ('PERSON',):
                    healers.append(ent.text)
            # treatments/symptoms: noun chunks and entities not person
            for chunk in doc.noun_chunks:
                ch = chunk.text.strip()
                # heuristics: if chunk contains words like 'tea', 'poultice', 'bark', treat as treatment
                if re.search(r'\b(tea|poultice|bark|tincture|herb|honey|saltwater|ointment|pills|crushed)\b', ch, re.I):
                    treatments.append(ch)
                else:
                    # otherwise could be symptom
                    # symptoms often include words like fever, cough, infection
                    if re.search(r'\b(fever|cough|infection|wound|stomach|ache|inflammation|sleeplessness|sleep)\b', ch, re.I):
                        symptoms.append(ch)
            # diseases: look for disease-like tokens
            for ent in doc.ents:
                if ent.label_ in ('DISEASE', 'CONDITION'):
                    diseases.append(ent.text)
        except Exception:
            pass

    # fallback heuristics from text
    if not treatments or not symptoms:
        words = text.lower()
        # common treatment words
        for tw in ['garlic', 'willow', 'honey', 'saltwater', 'mint', 'chamomile', 'poultice', 'herb', 'bark', 'tea']:
            if tw in words and tw not in treatments:
                treatments.append(tw)
        for sw in ['fever', 'cough', 'infection', 'wound', 'stomach', 'ache', 'inflammation', 'sleeplessness']:
            if sw in words and sw not in symptoms:
                symptoms.append(sw)

    # normalize lists
    def norm(lst):
        seen = []
        for v in lst:
            v2 = re.sub(r"[^a-z0-9\s'-]", '', v.lower()).strip()
            if v2 and v2 not in seen:
                seen.append(v2)
        return seen

    return {
        'healers': norm(healers),
        'treatments': norm(treatments),
        'symptoms': norm(symptoms),
        'diseases': norm(diseases),
    }


def classify_record(rec: Dict[str, Any]) -> str:
    """Rule-based classification of a parsed record into labels:
    'effective', 'failure', 'complaint', 'praise', 'neutral'.
    """
    outcome = (rec.get('outcome') or '').lower()
    sentiment = rec.get('sentiment', 'neutral')
    # explicit failure terms
    if any(x in outcome for x in ['did not', "didn't", 'no improvement', 'failed', 'worse', 'ineffective', 'no help', 'nothing changed']):
        return 'failure'
    # explicit praise
    if any(x in outcome for x in ['worked', 'healed', 'improved', 'patients improved', 'helped', 'recovered', 'broke']):
        return 'effective'
    # sentiment-based mapping
    if sentiment == 'positive':
        return 'effective'
    if sentiment == 'negative':
        return 'failure'
    # complaints mention 'complain' or 'complaint'
    if 'complain' in outcome or 'complaint' in outcome:
        return 'complaint'
    # praise generic
    if 'praise' in outcome:
        return 'praise'
    return 'neutral'


def topics_from_texts(texts: List[str], top_n: int = 5) -> List[str]:
    """Return top topic keywords using TF-IDF when available, else frequency."""
    if not texts:
        return []
    joined = ' '.join(texts)
    # prefer TF-IDF top words
    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
            X = vectorizer.fit_transform(texts)
            # sum scores and pick top features
            scores = X.sum(axis=0).A1
            indices = scores.argsort()[-top_n:][::-1]
            return [vectorizer.get_feature_names_out()[i] for i in indices]
        except Exception:
            pass
    # fallback: most common words longer than 3
    from collections import Counter
    words = re.findall(r"[a-zA-Z]{4,}", joined.lower())
    ctr = Counter(words)
    return [w for w, _ in ctr.most_common(top_n)]


def find_similar_cases(query_text: str, all_records: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Find the top N most similar cases to the query text using cosine similarity.
    
    Uses TF-IDF vectorization and cosine similarity when sklearn is available,
    otherwise falls back to simple keyword overlap matching.
    """
    if not all_records or not query_text:
        return []
    
    if SKLEARN_AVAILABLE:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            # Prepare texts: query + all record raw texts
            texts = [query_text] + [r.get('raw', '') for r in all_records]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            vectors = vectorizer.fit_transform(texts)
            # Compute similarity between query (first vector) and all records
            similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            # Get top N indices
            top_indices = similarities.argsort()[-top_n:][::-1]
            results = []
            for idx in top_indices:
                rec_copy = all_records[idx].copy()
                rec_copy['similarity_score'] = float(similarities[idx])
                results.append(rec_copy)
            return results
        except Exception:
            pass
    
    # Fallback: simple keyword overlap
    from collections import Counter
    query_words = set(re.findall(r"[a-zA-Z]{3,}", query_text.lower()))
    scores = []
    for rec in all_records:
        raw = rec.get('raw', '')
        rec_words = set(re.findall(r"[a-zA-Z]{3,}", raw.lower()))
        overlap = len(query_words & rec_words)
        scores.append(overlap)
    
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    results = []
    for idx in top_indices:
        rec_copy = all_records[idx].copy()
        rec_copy['similarity_score'] = scores[idx] / max(len(query_words), 1)
        results.append(rec_copy)
    return results


def answer_question(question: str, records: List[Dict[str, Any]]) -> str:
    """Answer a question about cures using the extracted records.
    
    Uses simple keyword matching and sentiment filtering to find relevant answers.
    """
    if not records or not question:
        return "No data available to answer the question."
    
    question_lower = question.lower()
    
    # Extract symptom from question
    symptom_keywords = ['fever', 'cough', 'infection', 'wound', 'stomach', 'ache', 'inflammation', 'sleeplessness', 'pain']
    detected_symptom = None
    for sym in symptom_keywords:
        if sym in question_lower:
            detected_symptom = sym
            break
    
    # Filter records by symptom if detected
    relevant_records = records
    if detected_symptom:
        relevant_records = [r for r in records if detected_symptom in r.get('symptom', '').lower() or detected_symptom in r.get('raw', '').lower()]
    
    # Check if asking for best/effective cures
    if any(word in question_lower for word in ['best', 'effective', 'work', 'good', 'help']):
        positive_records = [r for r in relevant_records if r.get('sentiment') == 'positive']
        if positive_records:
            # Count cures
            cure_counts = {}
            for r in positive_records:
                cure = r.get('cure', '').strip()
                if cure:
                    cure_counts[cure] = cure_counts.get(cure, 0) + 1
            if cure_counts:
                top_cure = max(cure_counts.items(), key=lambda x: x[1])
                if detected_symptom:
                    return f"Best cure for {detected_symptom}: {top_cure[0]} (mentioned {top_cure[1]} time{'s' if top_cure[1] > 1 else ''} positively)"
                else:
                    return f"Most effective cure: {top_cure[0]} (mentioned {top_cure[1]} time{'s' if top_cure[1] > 1 else ''} positively)"
    
    # Check if asking for worst/failed cures
    if any(word in question_lower for word in ['worst', 'fail', 'ineffective', 'bad', 'not work']):
        negative_records = [r for r in relevant_records if r.get('sentiment') == 'negative']
        if negative_records:
            cure_counts = {}
            for r in negative_records:
                cure = r.get('cure', '').strip()
                if cure:
                    cure_counts[cure] = cure_counts.get(cure, 0) + 1
            if cure_counts:
                worst_cure = max(cure_counts.items(), key=lambda x: x[1])
                if detected_symptom:
                    return f"Most failed cure for {detected_symptom}: {worst_cure[0]} (mentioned {worst_cure[1]} time{'s' if worst_cure[1] > 1 else ''} negatively)"
                else:
                    return f"Most failed cure: {worst_cure[0]} (mentioned {worst_cure[1]} time{'s' if worst_cure[1] > 1 else ''} negatively)"
    
    # General question - return summary
    if relevant_records and relevant_records != records:
        pos_cures = [r.get('cure') for r in relevant_records if r.get('sentiment') == 'positive' and r.get('cure')]
        neg_cures = [r.get('cure') for r in relevant_records if r.get('sentiment') == 'negative' and r.get('cure')]
        parts = []
        if pos_cures:
            parts.append(f"Effective cures for {detected_symptom}: {', '.join(set(pos_cures))}")
        if neg_cures:
            parts.append(f"Ineffective: {', '.join(set(neg_cures))}")
        if parts:
            return '. '.join(parts)
    
    return "No clear answer found. Try asking about specific symptoms (fever, cough, infection) or best/worst cures."


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

    keywords = extract_keywords(texts_for_k, top_n=12)

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

    # Extract entities and classify records
    entities = extract_entities(text)
    classified_records = []
    for rec in records:
        rec_copy = rec.copy()
        rec_copy['classification'] = classify_record(rec)
        classified_records.append(rec_copy)
    
    # Generate topics
    raw_texts = [r.get('raw', '') for r in records]
    topics = topics_from_texts(raw_texts, top_n=5) if raw_texts else []

    result = {
        'records': classified_records,
        'cures_pos_counts': cures_pos,
        'cures_neg_counts': cures_neg,
        'keywords': keywords,
        'summary': summary,
        'sentiment_scores': sentiment_scores,
        'entities': entities,
        'topics': topics,
    }

    # If spaCy is available, attempt to refine and normalize records (lemmatize cures/symptoms)
    try:
        if SPACY_AVAILABLE and _nlp is not None and records:
            for i, rec in enumerate(result['records']):
                raw = rec.get('raw', '')
                try:
                    doc = _nlp(raw)
                    # try to find direct object / noun chunk after a verb like 'use', 'apply', 'try'
                    cure_candidate = ''
                    symptom_candidate = ''
                    for token in doc:
                        if token.lemma_.lower() in ('use', 'apply', 'try', 'tried', 'used', 'applied') and token.i < len(doc) - 1:
                            # look for noun chunks that start after this token
                            for chunk in doc.noun_chunks:
                                if chunk.start >= token.i:
                                    cure_candidate = chunk.lemma_.lower().strip()
                                    break
                            if cure_candidate:
                                break

                    # also look for prepositional 'for' to capture symptom
                    for token in doc:
                        if token.text.lower() == 'for' and token.i < len(doc) - 1:
                            # take the noun chunk that contains the following token
                            for chunk in doc.noun_chunks:
                                if chunk.start <= token.i + 1 <= chunk.end:
                                    symptom_candidate = chunk.lemma_.lower().strip()
                                    break
                            if symptom_candidate:
                                break

                    # fallback: entities
                    if not cure_candidate:
                        ents = [ent.lemma_.lower().strip() for ent in doc.ents if len(ent.lemma_) > 2]
                        if ents:
                            cure_candidate = ents[0]

                    # apply normalized values if they look reasonable
                    def clean_val(v: str) -> str:
                        return re.sub(r"[^a-z0-9\s'-]", '', (v or '').strip()).lower()

                    if cure_candidate:
                        nv = clean_val(cure_candidate)
                        if nv:
                            result['records'][i]['cure'] = nv
                    if symptom_candidate:
                        nv2 = clean_val(symptom_candidate)
                        if nv2:
                            result['records'][i]['symptom'] = nv2
                except Exception:
                    # non-fatal; keep original
                    continue
            # recompute counts with normalized cures
            cures_pos = {}
            cures_neg = {}
            for r in result['records']:
                cure = (r.get('cure') or '').strip()
                s = r.get('sentiment')
                if not cure:
                    continue
                if s == 'positive':
                    cures_pos[cure] = cures_pos.get(cure, 0) + 1
                elif s == 'negative':
                    cures_neg[cure] = cures_neg.get(cure, 0) + 1
            result['cures_pos_counts'] = cures_pos
            result['cures_neg_counts'] = cures_neg
    except Exception:
        # keep original result on any failure
        pass

    return result


if __name__ == '__main__':
    sample = "Healer Anna used garlic for infections — patients healed quickly. Healer John used saltwater for fever — it didn't help."
    print(process_scrolls(sample))
