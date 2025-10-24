# The Healer's Scribe — Prototype

Lightweight Streamlit prototype that parses messy healer notes and extracts cures, symptoms, and outcomes.

Prerequisites
- Python 3.8+

# Quick start (PowerShell)

```powershell
# (optional) create venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install -r requirements.txt

# Optional: install extras for better NLP
python -m pip install spacy nltk scikit-learn transformers
python -m spacy download en_core_web_sm

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
- Note: the app was converted to a Flask app; see instructions above to run.
- `sample_input.txt` — example healer notes
- `test_run.py` — small script to run parser and print results

Notes
- This is a fast hackathon prototype using heuristics. Replace or extend `nlp.parse_text` with spaCy/HuggingFace models for production.

File inputs & exports
- The app accepts pasted text or file uploads (PDF, JSON, TXT). If you upload a PDF the server will extract text using `PyPDF2` (included in `requirements.txt`).
- Export options: CSV (records), JSON (full result dict), TXT (plain text summary + records), and PDF (generated server-side using `fpdf`). If you want PDF export, install `fpdf` (already listed in `requirements.txt`).

6-hour Hackathon Plan (summary)

This project is organized to be completed in a 6-hour sprint. Key milestones:

Hour 1 — Project scaffolding
- Create Flask app, templates and static files.

Hour 2 — Input/output pages
- Build `index.html` and `result.html` pages to accept input and show results.

Hour 3–4 — AI core
- Implement `models/nlp_pipeline.py` (NER, sentiment, TF-IDF, summarization fallbacks).

Hour 5 — Connect Flask
- Wire routes to processing and add download endpoints (CSV/JSON/TXT/PDF).

Hour 6 — Polish and present
- Add charts, styling, and package the folder for deployment or demo.

Files included in this repo
- `app.py` — Flask server and routes (index, downloads)
- `models/nlp_pipeline.py` — processing wrapper (uses heuristics or optional heavy libs)
- `nlp.py` — simple rule-based parser used as fallback
- `templates/` — `index.html` (main UI) and `result.html` (result layout)
- `static/` — `styles.css` and `chart.js` for UI polish
- `requirements.txt` — minimal deps; extras optional for better NLP

Want a ready package? Run the `package_project.ps1` script (added to repo) or use the PowerShell compress command to build a zip of the project.
