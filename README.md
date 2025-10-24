# The Healer's Scribe — Prototype

Lightweight Streamlit prototype that parses messy healer notes and extracts cures, symptoms, and outcomes.

Prerequisites
- Python 3.8+

Quick start (PowerShell)

```powershell
# (optional) create venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install -r requirements.txt

# run the Flask app
$env:FLASK_APP = 'app'
flask run
```

Run the quick parser unit (no Flask needed):

```powershell
python test_run.py
```

What is included
- `nlp.py` — rule-based parser and sentiment heuristics
- `app.py` — Streamlit UI and visualizations
- `sample_input.txt` — example healer notes
- `test_run.py` — small script to run parser and print results

Notes
- This is a fast hackathon prototype using heuristics. Replace or extend `nlp.parse_text` with spaCy/HuggingFace models for production.
