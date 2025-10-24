from flask import Flask, render_template, request, make_response, jsonify
from models.nlp_pipeline import process_scrolls
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
from werkzeug.utils import secure_filename
import tempfile
import pathlib

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

app = Flask(__name__)

SAMPLE_TEXT = """
Healer A used herb willow for fever, it worked well.
Healer B used honey for cough, patients improved.
Healer C tried willow for infection but results were poor.
Healer Anna used garlic for infections — patients healed quickly.
Healer John used saltwater for fever — it didn't help.
"""


# Word cloud removed — server-side word cloud generation was removed per request.


@app.route('/app', methods=['GET', 'POST'])
def index():
    text = SAMPLE_TEXT
    table_html = None
    pos_div = None
    neg_div = None
    summary = None

    if request.method == 'POST':
        # accept textarea or file upload (pdf, json, txt)
        uploaded = request.files.get('file')
        text = request.form.get('text', '')
        if uploaded and uploaded.filename:
            filename = secure_filename(uploaded.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.pdf' and PYPDF2_AVAILABLE:
                try:
                    reader = PdfReader(uploaded.stream)
                    pages = [p.extract_text() or '' for p in reader.pages]
                    text = '\n'.join(pages)
                except Exception:
                    text = uploaded.stream.read().decode('utf-8', errors='ignore')
            elif ext == '.json':
                try:
                    j = json.load(uploaded.stream)
                    # if it's a list of strings
                    if isinstance(j, list):
                        text = '\n'.join([str(x) for x in j])
                    elif isinstance(j, dict):
                        # join values
                        text = '\n'.join([str(v) for v in j.values()])
                    else:
                        text = str(j)
                except Exception:
                    text = uploaded.stream.read().decode('utf-8', errors='ignore')
            else:
                # treat as text
                text = uploaded.stream.read().decode('utf-8', errors='ignore')

        # Process text and render results page
        result = process_scrolls(text)
        result.setdefault('records', [])
        result.setdefault('cures_pos_counts', {})
        result.setdefault('cures_neg_counts', {})
        result.setdefault('keywords', [])
        result.setdefault('summary', '')

        # compute percent effectiveness and insight one-liner
        pos = result.get('cures_pos_counts', {})
        neg = result.get('cures_neg_counts', {})
        all_cures = set(list(pos.keys()) + list(neg.keys()))
        effectiveness = {}
        for c in all_cures:
            p = int(pos.get(c, 0))
            n = int(neg.get(c, 0))
            total = p + n
            pct = int((p / total) * 100) if total > 0 else 0
            effectiveness[c] = {'pos': p, 'neg': n, 'total': total, 'pct': pct}

        # strongest cure: highest pct (require at least 1 total), tie-breaker by pos count
        strongest = None
        most_failed = None
        if effectiveness:
            strongest = max(all_cures, key=lambda k: (effectiveness[k]['pct'], effectiveness[k]['pos']))
            most_failed = max(all_cures, key=lambda k: (effectiveness[k]['neg'], effectiveness[k]['total']))

        insight_parts = []
        if strongest and effectiveness[strongest]['total'] > 0:
            insight_parts.append(f"Strongest cure: {strongest} ({effectiveness[strongest]['pct']}% effective)")
        if most_failed and effectiveness[most_failed]['neg'] > 0:
            insight_parts.append(f"Most failed cure: {most_failed} ({effectiveness[most_failed]['neg']} failures)")
        result['insight'] = ' • '.join(insight_parts) if insight_parts else result.get('summary','')

        # Create Plotly bar charts server-side
        try:
            # Positive chart
            pos_items = sorted(pos.items(), key=lambda x: x[1], reverse=True)
            pos_names = [k for k, v in pos_items]
            pos_vals = [v for k, v in pos_items]
            pos_text = [f"{effectiveness.get(n,{}).get('pct',0)}%" for n in pos_names]
            pos_fig = go.Figure(go.Bar(x=pos_names, y=pos_vals, text=pos_text, textposition='auto', marker_color='seagreen'))
            pos_fig.update_layout(title='Top Effective Cures', xaxis_title='Cure', yaxis_title='Positive reports')
            result['pos_chart_div'] = pio.to_html(pos_fig, full_html=False, include_plotlyjs='cdn')

            # Negative chart
            neg_items = sorted(neg.items(), key=lambda x: x[1], reverse=True)
            neg_names = [k for k, v in neg_items]
            neg_vals = [v for k, v in neg_items]
            neg_text = [f"{effectiveness.get(n,{}).get('pct',0)}%" for n in neg_names]
            neg_fig = go.Figure(go.Bar(x=neg_names, y=neg_vals, text=neg_text, textposition='auto', marker_color='indianred'))
            neg_fig.update_layout(title='Top Ineffective Cures', xaxis_title='Cure', yaxis_title='Negative reports')
            result['neg_chart_div'] = pio.to_html(neg_fig, full_html=False, include_plotlyjs='cdn')
        except Exception:
            # fallback: ensure keys exist
            result.setdefault('pos_chart_div', '')
            result.setdefault('neg_chart_div', '')

        result['original_text'] = text
        return render_template('result.html', result=result)

    # For GET requests render the input form
    return render_template('index.html', text=text, table_html=table_html, pos_div=pos_div, neg_div=neg_div, summary=summary)


@app.route('/')
def landing():
    return render_template('landing.html')


# Lightweight JSON health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


# JSON API: synchronous processing endpoint
@app.route('/api/process', methods=['POST'])
def api_process():
    """Accept JSON or form data with a 'text' field and return JSON processing result.

    Example JSON: {"text": "Healer A used garlic for infection, it worked."}
    """
    # extract text from JSON body or form-data or raw body
    text = None
    if request.is_json:
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            text = data.get('text')
    else:
        # form data
        text = request.form.get('text')
        # fallback to raw body
        if not text and request.data:
            try:
                text = request.data.decode('utf-8')
            except Exception:
                text = None

    if not text or not str(text).strip():
        return make_response((json.dumps({'error': 'no text provided'}), 400, {'Content-Type': 'application/json'}))

    try:
        result = process_scrolls(text)
        # ensure keys exist for stable clients
        result.setdefault('records', [])
        result.setdefault('cures_pos_counts', {})
        result.setdefault('cures_neg_counts', {})
        result.setdefault('keywords', [])
        result.setdefault('summary', '')
        result.setdefault('entities', {})
        result.setdefault('topics', [])
        result['original_text'] = text
        # return JSON
        return make_response((json.dumps(result, ensure_ascii=False), 200, {'Content-Type': 'application/json'}))
    except Exception as e:
        app.logger.exception('Processing failed')
        return make_response((json.dumps({'error': 'processing_failed'}), 500, {'Content-Type': 'application/json'}))


@app.route('/api/similar', methods=['POST'])
def api_similar():
    """Find similar cases to a given query text.
    
    Example JSON: {"query": "used garlic for infection", "text": "full healer notes..."}
    """
    from models.nlp_pipeline import find_similar_cases
    
    data = request.get_json(silent=True) if request.is_json else {}
    query = data.get('query') or request.form.get('query')
    text = data.get('text') or request.form.get('text')
    top_n = int(data.get('top_n', 3))
    
    if not query:
        return make_response((json.dumps({'error': 'no query provided'}), 400, {'Content-Type': 'application/json'}))
    
    if not text:
        return make_response((json.dumps({'error': 'no text data provided'}), 400, {'Content-Type': 'application/json'}))
    
    try:
        # Process the text to get records
        result = process_scrolls(text)
        records = result.get('records', [])
        
        # Find similar cases
        similar = find_similar_cases(query, records, top_n=top_n)
        
        response = {
            'query': query,
            'similar_cases': similar,
            'count': len(similar)
        }
        return make_response((json.dumps(response, ensure_ascii=False), 200, {'Content-Type': 'application/json'}))
    except Exception as e:
        app.logger.exception('Similar case search failed')
        return make_response((json.dumps({'error': 'search_failed'}), 500, {'Content-Type': 'application/json'}))


@app.route('/ask-rag', methods=['POST'])
def ask_rag():
    # Enhanced Q&A endpoint using extracted knowledge
    q = request.form.get('question') or request.json.get('question') if request.is_json else None
    text = request.form.get('text') or request.json.get('text') if request.is_json else None
    
    if not q:
        return make_response((json.dumps({'error': 'no question provided'}), 400, {'Content-Type':'application/json'}))
    
    # If text is provided, process it and answer the question
    if text:
        from models.nlp_pipeline import answer_question
        result = process_scrolls(text)
        records = result.get('records', [])
        answer_text = answer_question(q, records)
    else:
        answer_text = "No text data provided. Please include 'text' field with healer notes to analyze."
    
    answer = {
        'answer': answer_text,
        'question': q
    }
    return make_response((json.dumps(answer, ensure_ascii=False), 200, {'Content-Type':'application/json'}))
    


@app.route('/analyze', methods=['POST'])
def analyze():
    # Compare Healers panel posts here (compare_a, compare_b). Render results at /analyze so URL is bookmarkable.
    a = request.form.get('compare_a', '')
    b = request.form.get('compare_b', '')
    combined = (a or '') + '\n\n----\n\n' + (b or '')
    text = combined.strip() or request.form.get('text','')
    result = process_scrolls(text)
    result.setdefault('records', [])
    result.setdefault('cures_pos_counts', {})
    result.setdefault('cures_neg_counts', {})
    result.setdefault('keywords', [])
    result.setdefault('summary', '')

    # compute insight + charts (same logic as index)
    pos = result.get('cures_pos_counts', {})
    neg = result.get('cures_neg_counts', {})
    all_cures = set(list(pos.keys()) + list(neg.keys()))
    effectiveness = {}
    for c in all_cures:
        p = int(pos.get(c, 0))
        n = int(neg.get(c, 0))
        total = p + n
        pct = int((p / total) * 100) if total > 0 else 0
        effectiveness[c] = {'pos': p, 'neg': n, 'total': total, 'pct': pct}
    strongest = None
    most_failed = None
    if effectiveness:
        strongest = max(all_cures, key=lambda k: (effectiveness[k]['pct'], effectiveness[k]['pos']))
        most_failed = max(all_cures, key=lambda k: (effectiveness[k]['neg'], effectiveness[k]['total']))
    insight_parts = []
    if strongest and effectiveness[strongest]['total'] > 0:
        insight_parts.append(f"Strongest cure: {strongest} ({effectiveness[strongest]['pct']}% effective)")
    if most_failed and effectiveness[most_failed]['neg'] > 0:
        insight_parts.append(f"Most failed cure: {most_failed} ({effectiveness[most_failed]['neg']} failures)")
    result['insight'] = ' • '.join(insight_parts) if insight_parts else result.get('summary','')

    try:
        pos_items = sorted(pos.items(), key=lambda x: x[1], reverse=True)
        pos_names = [k for k, v in pos_items]
        pos_vals = [v for k, v in pos_items]
        pos_text = [f"{effectiveness.get(n,{}).get('pct',0)}%" for n in pos_names]
        pos_fig = go.Figure(go.Bar(x=pos_names, y=pos_vals, text=pos_text, textposition='auto', marker_color='seagreen'))
        pos_fig.update_layout(title='Top Effective Cures', xaxis_title='Cure', yaxis_title='Positive reports')
        result['pos_chart_div'] = pio.to_html(pos_fig, full_html=False, include_plotlyjs='cdn')

        neg_items = sorted(neg.items(), key=lambda x: x[1], reverse=True)
        neg_names = [k for k, v in neg_items]
        neg_vals = [v for k, v in neg_items]
        neg_text = [f"{effectiveness.get(n,{}).get('pct',0)}%" for n in neg_names]
        neg_fig = go.Figure(go.Bar(x=neg_names, y=neg_vals, text=neg_text, textposition='auto', marker_color='indianred'))
        neg_fig.update_layout(title='Top Ineffective Cures', xaxis_title='Cure', yaxis_title='Negative reports')
        result['neg_chart_div'] = pio.to_html(neg_fig, full_html=False, include_plotlyjs='cdn')
    except Exception:
        result.setdefault('pos_chart_div', '')
        result.setdefault('neg_chart_div', '')

    result['original_text'] = text
    return render_template('result.html', result=result)


@app.route('/download', methods=['POST'])
def download():
    text = request.form.get('text', '')
    # Re-run processing and return CSV
    result = process_scrolls(text)
    records = result.get('records', [])
    df = pd.DataFrame(records)
    csv = df.to_csv(index=False)
    resp = make_response(csv)
    resp.headers['Content-Disposition'] = 'attachment; filename=healers_results.csv'
    resp.mimetype = 'text/csv'
    return resp


@app.route('/download/json', methods=['POST'])
def download_json():
    text = request.form.get('text', '')
    result = process_scrolls(text)
    data = json.dumps(result, ensure_ascii=False, indent=2)
    resp = make_response(data)
    resp.headers['Content-Disposition'] = 'attachment; filename=healers_results.json'
    resp.mimetype = 'application/json'
    return resp


@app.route('/download/txt', methods=['POST'])
def download_txt():
    text = request.form.get('text', '')
    result = process_scrolls(text)
    lines = []
    lines.append('Summary:')
    lines.append(result.get('summary',''))
    lines.append('\nRecords:')
    for r in result.get('records', []):
        lines.append(f"Healer: {r.get('healer')} | Cure: {r.get('cure')} | Symptom: {r.get('symptom')} | Outcome: {r.get('outcome')} | Sentiment: {r.get('sentiment')}")
    data = '\n'.join(lines)
    resp = make_response(data)
    resp.headers['Content-Disposition'] = 'attachment; filename=healers_results.txt'
    resp.mimetype = 'text/plain'
    return resp


@app.route('/download/pdf', methods=['POST'])
def download_pdf():
    text = request.form.get('text', '')
    result = process_scrolls(text)

    if not FPDF_AVAILABLE:
        # fallback to TXT download if fpdf is not installed
        lines = ["Wisdom Scroll (plain text)", "", f"Insight: {result.get('insight', result.get('summary',''))}", '', 'Records:']
        for r in result.get('records', []):
            lines.append(f"Healer: {r.get('healer')} | Cure: {r.get('cure')} | Symptom: {r.get('symptom')} | Outcome: {r.get('outcome')} | Sentiment: {r.get('sentiment')}")
        data = '\n'.join(lines)
        resp = make_response(data)
        resp.headers['Content-Disposition'] = 'attachment; filename=healers_results.txt'
        resp.mimetype = 'text/plain'
        return resp

    # Try to generate images for charts if plotly is available
    pos_png = None
    neg_png = None
    try:
        pos_counts = result.get('cures_pos_counts', {})
        neg_counts = result.get('cures_neg_counts', {})
        if pos_counts:
            pos_items = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
            pos_names = [k for k, v in pos_items]
            pos_vals = [v for k, v in pos_items]
            pos_fig = go.Figure(go.Bar(x=pos_names, y=pos_vals, marker_color='seagreen'))
            pos_fig.update_layout(title='Top Effective Cures', xaxis_title='Cure', yaxis_title='Positive reports')
            pos_png = pio.to_image(pos_fig, format='png')
        if neg_counts:
            neg_items = sorted(neg_counts.items(), key=lambda x: x[1], reverse=True)
            neg_names = [k for k, v in neg_items]
            neg_vals = [v for k, v in neg_items]
            neg_fig = go.Figure(go.Bar(x=neg_names, y=neg_vals, marker_color='indianred'))
            neg_fig.update_layout(title='Top Ineffective Cures', xaxis_title='Cure', yaxis_title='Negative reports')
            neg_png = pio.to_image(neg_fig, format='png')
    except Exception:
        pos_png = None
        neg_png = None

    # Build PDF using FPDF, include images if we managed to render them
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font('Times', 'B', 18)
    pdf.cell(0, 10, "🧙‍♀️ The Healer's Scribe — Wisdom Scroll", ln=True, align='C')
    pdf.ln(4)
    pdf.set_font('Times', '', 12)
    pdf.multi_cell(0, 6, f"Insight: {result.get('insight', result.get('summary',''))}")
    pdf.ln(6)

    temp_files = []
    try:
        # Add positive chart image
        if pos_png:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            tf.write(pos_png)
            tf.flush()
            tf.close()
            temp_files.append(tf.name)
            pdf.image(tf.name, w=180)
            pdf.ln(4)

        # Add negative chart image
        if neg_png:
            tf2 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            tf2.write(neg_png)
            tf2.flush()
            tf2.close()
            temp_files.append(tf2.name)
            pdf.image(tf2.name, w=180)
            pdf.ln(6)

        # Records table (compact)
        pdf.set_font('Times', 'B', 13)
        pdf.cell(0, 8, 'Records', ln=True)
        pdf.set_font('Times', '', 10)
        for r in result.get('records', []):
            line = f"Healer: {r.get('healer')} | Cure: {r.get('cure')} | Outcome: {r.get('outcome')} | Sentiment: {r.get('sentiment')}"
            pdf.multi_cell(0, 6, line)

        out = pdf.output(dest='S').encode('latin-1')
        resp = make_response(out)
        resp.headers['Content-Disposition'] = 'attachment; filename=healers_wisdom_scroll.pdf'
        resp.mimetype = 'application/pdf'
        return resp
    finally:
        # clean up temp files
        for p in temp_files:
            try:
                pathlib.Path(p).unlink()
            except Exception:
                pass


if __name__ == '__main__':
    app.run(debug=True)

