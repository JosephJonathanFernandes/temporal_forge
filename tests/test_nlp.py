from models.nlp_pipeline import process_scrolls
from models.nlp_pipeline import process_scrolls


def test_process_scrolls_basic():
    text = (
        "Healer A used garlic for infection, it worked well. "
        "Healer B used saltwater for fever - it did not help."
    )
    res = process_scrolls(text)
    assert 'records' in res
    assert isinstance(res['records'], list)
    # expect garlic positive and saltwater negative (depending on heuristics)
    pos = res.get('cures_pos_counts', {})
    neg = res.get('cures_neg_counts', {})
    # at least one positive cure should be present
    assert any(v > 0 for v in pos.values())
    assert isinstance(res.get('keywords', []), list)


def test_process_scrolls_sample_file():
    with open('sample_input.txt', 'r', encoding='utf-8') as f:
        s = f.read()
    res = process_scrolls(s)
    assert 'summary' in res
    # keywords should not be empty
    assert len(res.get('keywords', [])) > 0
