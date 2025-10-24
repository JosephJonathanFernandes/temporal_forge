from flask import Flask, render_template, request, make_response
from models.nlp_pipeline import process_scrolls
import pandas as pd
import plotly.express as px

app = Flask(__name__)

SAMPLE_TEXT = """
Healer A used herb willow for fever, it worked well.
Healer B used honey for cough, patients improved.
Healer C tried willow for infection but results were poor.
Healer Anna used garlic for infections — patients healed quickly.
Healer John used saltwater for fever — it didn't help.
"""


# Word cloud removed — server-side word cloud generation was removed per request.


@app.route('/', methods=['GET', 'POST'])
def index():
    text = SAMPLE_TEXT
    table_html = None
    pos_div = None
    neg_div = None
    summary = None

    if request.method == 'POST':
        text = request.form.get('text', '')
        result = process_scrolls(text)
        records = result.get('records', [])
        if records:
            df = pd.DataFrame(records)
            table_html = df[['healer', 'cure', 'symptom', 'outcome', 'sentiment']].to_html(classes='table table-sm', index=False, escape=False)

            cures_pos = pd.Series(result.get('cures_pos_counts', {})).reset_index()
            cures_pos.columns = ['cure', 'count']
            cures_neg = pd.Series(result.get('cures_neg_counts', {})).reset_index()
            cures_neg.columns = ['cure', 'count']

            if not cures_pos.empty:
                fig_pos = px.bar(cures_pos.head(10), x='cure', y='count', title='Top Effective Cures')
                pos_div = fig_pos.to_html(full_html=False, include_plotlyjs='cdn')
            if not cures_neg.empty:
                fig_neg = px.bar(cures_neg.head(10), x='cure', y='count', title='Top Ineffective Cures')
                neg_div = fig_neg.to_html(full_html=False, include_plotlyjs='cdn')

            summary = result.get('summary')

    return render_template('index.html', text=text, table_html=table_html, pos_div=pos_div, neg_div=neg_div, summary=summary)


@app.route('/download', methods=['POST'])
def download():
    text = request.form.get('text', '')
    result = process_scrolls(text)
    records = result.get('records', [])
    df = pd.DataFrame(records)
    csv = df.to_csv(index=False)
    resp = make_response(csv)
    resp.headers['Content-Disposition'] = 'attachment; filename=healers_results.csv'
    resp.mimetype = 'text/csv'
    return resp


if __name__ == '__main__':
    app.run(debug=True)

