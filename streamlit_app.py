import streamlit as st
import pickle

st.set_page_config(page_title="Sentiment Analyzer", page_icon="◈", layout="centered")

@st.cache_resource
def load_model():
    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #f5f7fa;
    color: #1a1a2e;
}

.main .block-container {
    padding: 2.5rem 2rem 4rem 2rem;
    max-width: 700px;
}

#MainMenu, footer, header { visibility: hidden; }

/* Header */
.hero-wrap {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 0.85rem;
    color: #8892b0;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    color: #a8b2d8;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Card */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #eef0f5;
    border-radius: 10px;
    padding: 4px;
    border: none;
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #6b7280;
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 0.55rem 1.5rem;
}

.stTabs [aria-selected="true"] {
    color: #1a1a2e !important;
    background: #ffffff !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}

/* Textarea */
.stTextArea textarea {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.9rem 1rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s !important;
}

.stTextArea textarea:focus {
    border-color: #0f3460 !important;
    box-shadow: 0 0 0 3px rgba(15,52,96,0.08) !important;
    background: #fff !important;
}

.stTextArea textarea::placeholder { color: #adb5bd !important; }
.stTextArea label { color: #374151 !important; font-weight: 600 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1a1a2e, #0f3460) !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    margin-top: 0.75rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Result card */
.result-wrap {
    margin-top: 1.5rem;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

.result-header {
    padding: 1.75rem 2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.result-emoji { font-size: 2.5rem; line-height: 1; }

.result-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.result-conf {
    font-size: 0.8rem;
    font-weight: 500;
    opacity: 0.75;
    margin-top: 0.2rem;
}

.result-body {
    background: #ffffff;
    padding: 1.5rem 2rem;
}

.score-label-text {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6b7280;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.score-bar-bg {
    background: #f1f5f9;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 0.3rem;
}

.score-bar-fill {
    height: 100%;
    border-radius: 6px;
}

.score-pct-text {
    font-size: 0.75rem;
    font-weight: 600;
    color: #374151;
    text-align: right;
}

/* positive */
.theme-positive .result-header { background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; }
.theme-positive .result-label  { color: #065f46; }
.theme-positive .bar-fill      { background: #10b981; }

/* negative */
.theme-negative .result-header { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #7f1d1d; }
.theme-negative .result-label  { color: #7f1d1d; }
.theme-negative .bar-fill      { background: #ef4444; }

/* neutral */
.theme-neutral .result-header { background: linear-gradient(135deg, #e0e7ff, #c7d2fe); color: #1e1b4b; }
.theme-neutral .result-label  { color: #312e81; }
.theme-neutral .bar-fill      { background: #6366f1; }

/* Batch rows */
.batch-row {
    background: #ffffff;
    border: 1.5px solid #f1f5f9;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.batch-badge {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    flex-shrink: 0;
}

.badge-positive { background: #d1fae5; color: #065f46; }
.badge-negative { background: #fee2e2; color: #7f1d1d; }
.badge-neutral  { background: #e0e7ff; color: #312e81; }

.batch-text { font-size: 0.88rem; color: #374151; font-weight: 500; flex: 1; }
.batch-conf { font-size: 0.75rem; color: #9ca3af; font-weight: 500; flex-shrink: 0; }

.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #9ca3af;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
</style>

<div class="hero-wrap">
    <div class="hero-badge">◈ &nbsp; AI / ML Project</div>
    <div class="hero-title">Sentiment Analyzer</div>
    <div class="hero-sub">Classify text as Positive, Neutral, or Negative instantly</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.markdown('<p style="font-size:0.88rem; font-weight:600; color:#374151; margin-bottom:0.4rem;">Enter your text</p>', unsafe_allow_html=True)
    user_input = st.text_area("", placeholder="e.g. The product quality exceeded my expectations!", height=130, key="single_input", label_visibility="collapsed")

    if st.button("Analyze Sentiment →", key="single_btn"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            X = vectorizer.transform([user_input])
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 1)

            labels_map = {0: "Negative", 1: "Neutral",  2: "Positive"}
            emojis     = {0: "😞",        1: "😐",        2: "😊"}
            themes     = {0: "negative",  1: "neutral",  2: "positive"}
            bar_colors = {0: "#ef4444",   1: "#6366f1",  2: "#10b981"}

            neg = round(float(proba[0]) * 100, 1)
            neu = round(float(proba[1]) * 100, 1)
            pos = round(float(proba[2]) * 100, 1)

            def bar(pct, color):
                return f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;background:{color};"></div></div>'

            st.markdown(f"""
            <div class="result-wrap theme-{themes[pred]}">
                <div class="result-header">
                    <div class="result-emoji">{emojis[pred]}</div>
                    <div>
                        <div class="result-label">{labels_map[pred]}</div>
                        <div class="result-conf">{confidence}% confidence</div>
                    </div>
                </div>
                <div class="result-body">
                    <div class="score-label-text">Score Breakdown</div>
                    <table style="width:100%;border-collapse:collapse;">
                        <tr>
                            <td style="width:70px;font-size:0.75rem;font-weight:600;color:#6b7280;padding:4px 0;">Positive</td>
                            <td style="padding:4px 10px;">{bar(pos, '#10b981')}</td>
                            <td style="width:42px;font-size:0.75rem;font-weight:600;color:#374151;text-align:right;">{pos}%</td>
                        </tr>
                        <tr>
                            <td style="font-size:0.75rem;font-weight:600;color:#6b7280;padding:4px 0;">Neutral</td>
                            <td style="padding:4px 10px;">{bar(neu, '#6366f1')}</td>
                            <td style="font-size:0.75rem;font-weight:600;color:#374151;text-align:right;">{neu}%</td>
                        </tr>
                        <tr>
                            <td style="font-size:0.75rem;font-weight:600;color:#6b7280;padding:4px 0;">Negative</td>
                            <td style="padding:4px 10px;">{bar(neg, '#ef4444')}</td>
                            <td style="font-size:0.75rem;font-weight:600;color:#374151;text-align:right;">{neg}%</td>
                        </tr>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<p style="font-size:0.88rem; font-weight:600; color:#374151; margin-bottom:0.4rem;">Enter sentences (one per line)</p>', unsafe_allow_html=True)
    batch_input = st.text_area("", placeholder="I love this product!\nDelivery was on time.\nTerrible quality, very disappointed.", height=180, key="batch_input", label_visibility="collapsed")

    if st.button("Analyze All →", key="batch_btn"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one sentence.")
        else:
            X = vectorizer.transform(lines)
            preds = model.predict(X)
            probas = model.predict_proba(X)

            labels_map  = {0: "Negative", 1: "Neutral", 2: "Positive"}
            badge_class = {0: "badge-negative", 1: "badge-neutral", 2: "badge-positive"}

            rows_html = "".join([
                f"""<div class="batch-row">
                    <span class="batch-badge {badge_class[int(preds[i])]}">{labels_map[int(preds[i])]}</span>
                    <span class="batch-text">{text}</span>
                    <span class="batch-conf">{round(float(max(probas[i]))*100,1)}%</span>
                </div>"""
                for i, text in enumerate(lines)
            ])

            st.markdown(f"""
            <div style="margin-top:1.5rem;">
                <div class="section-label">{len(lines)} result{'s' if len(lines)>1 else ''}</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)
