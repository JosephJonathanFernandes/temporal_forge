from flask import Flask, render_template, request, make_response
from models.nlp_pipeline import process_scrolls
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

SAMPLE_TEXT = """
Healer A used herb willow for fever, it worked well.
Healer B used honey for cough, patients improved.
Healer C tried willow for infection but results were poor.
Healer Anna used garlic for infections — patients healed quickly.
Healer John used saltwater for fever — it didn't help.
"""


def make_wordcloud_image(text: str) -> str:
    if not text.strip():
        return ""
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"


@app.route('/', methods=['GET', 'POST'])
def index():
    text = SAMPLE_TEXT
    table_html = None
    pos_div = None
    neg_div = None
    wc_data = None
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

            text_for_wc = ' '.join(result.get('keywords', []))
            wc_data = make_wordcloud_image(text_for_wc)

            summary = result.get('summary')

    return render_template('index.html', text=text, table_html=table_html, pos_div=pos_div, neg_div=neg_div, wc_data=wc_data, summary=summary)


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

