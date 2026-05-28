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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.stApp {
    background: #0a0a0a;
    color: #f0f0f0;
}

.main .block-container {
    padding: 3rem 2rem 4rem 2rem;
    max-width: 720px;
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    color: #f0f0f0;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 3rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444;
    background: transparent;
    border: none;
    padding: 0.8rem 1.5rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}

.stTabs [aria-selected="true"] {
    color: #f0f0f0 !important;
    border-bottom: 2px solid #f0f0f0 !important;
    background: transparent !important;
}

/* Textarea */
.stTextArea textarea {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s;
}

.stTextArea textarea:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}

.stTextArea textarea::placeholder { color: #333 !important; }

/* Button */
.stButton > button {
    background: #f0f0f0 !important;
    color: #0a0a0a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    cursor: pointer !important;
    transition: background 0.2s, color 0.2s !important;
}

.stButton > button:hover {
    background: #ccc !important;
}

/* Result card */
.result-card {
    margin-top: 2.5rem;
    padding: 2.5rem;
    border: 1px solid #1f1f1f;
    border-radius: 4px;
    background: #0f0f0f;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.result-card.positive::before { background: #4ade80; }
.result-card.negative::before { background: #f87171; }
.result-card.neutral::before  { background: #94a3b8; }

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.result-label.positive { color: #4ade80; }
.result-label.negative { color: #f87171; }
.result-label.neutral  { color: #94a3b8; }

.result-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #555;
    width: 70px;
    flex-shrink: 0;
}

.score-bar-bg {
    flex: 1;
    height: 3px;
    background: #1a1a1a;
    border-radius: 2px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

.score-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* Batch result rows */
.batch-row {
    padding: 1rem 1.25rem;
    border: 1px solid #1a1a1a;
    border-radius: 3px;
    margin-bottom: 0.5rem;
    background: #0d0d0d;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.batch-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
    flex-shrink: 0;
    font-weight: 500;
}

.badge-positive { background: #052e16; color: #4ade80; }
.badge-negative { background: #2d0a0a; color: #f87171; }
.badge-neutral  { background: #0f172a; color: #94a3b8; }

.batch-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #888;
    flex: 1;
}

.batch-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #333;
    flex-shrink: 0;
}

.divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 2rem 0;
}
</style>

<div class="hero-title">Sentiment<br>Analyzer</div>
<div class="hero-sub">◈ &nbsp; NLP Classification Engine</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single", "Batch"])

with tab1:
    user_input = st.text_area("", placeholder="Type anything — a review, a message, a thought...", height=130, key="single_input")

    if st.button("ANALYZE", key="single_btn"):
        if not user_input.strip():
            st.warning("Enter some text first.")
        else:
            X = vectorizer.transform([user_input])
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 1)

            labels     = {0: "Negative", 1: "Neutral", 2: "Positive"}
            css_class  = {0: "negative", 1: "neutral", 2: "positive"}
            bar_colors = {0: "#f87171", 1: "#94a3b8", 2: "#4ade80"}

            neg_pct = round(float(proba[0]) * 100, 1)
            neu_pct = round(float(proba[1]) * 100, 1)
            pos_pct = round(float(proba[2]) * 100, 1)

            st.markdown(f"""
            <div class="result-card {css_class[pred]}">
                <div class="result-label {css_class[pred]}">{labels[pred]}</div>
                <div class="result-confidence">{confidence}% confidence</div>
                <div class="score-row">
                    <div class="score-label">Positive</div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pos_pct}%; background:#4ade80;"></div></div>
                    <div class="score-pct">{pos_pct}%</div>
                </div>
                <div class="score-row">
                    <div class="score-label">Neutral</div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{neu_pct}%; background:#94a3b8;"></div></div>
                    <div class="score-pct">{neu_pct}%</div>
                </div>
                <div class="score-row">
                    <div class="score-label">Negative</div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{neg_pct}%; background:#f87171;"></div></div>
                    <div class="score-pct">{neg_pct}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    batch_input = st.text_area("", placeholder="One sentence per line:\nI love this product!\nDelivery was on time.\nTerrible quality.", height=180, key="batch_input")

    if st.button("ANALYZE ALL", key="batch_btn"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Enter at least one sentence.")
        else:
            X = vectorizer.transform(lines)
            preds = model.predict(X)
            probas = model.predict_proba(X)

            labels_map  = {0: "Negative", 1: "Neutral", 2: "Positive"}
            badge_class = {0: "badge-negative", 1: "badge-neutral", 2: "badge-positive"}

            rows_html = ""
            for i, text in enumerate(lines):
                pred = int(preds[i])
                conf = round(float(max(probas[i])) * 100, 1)
                rows_html += f"""
                <div class="batch-row">
                    <span class="batch-badge {badge_class[pred]}">{labels_map[pred]}</span>
                    <span class="batch-text">{text}</span>
                    <span class="batch-conf">{conf}%</span>
                </div>"""

            st.markdown(f"""
            <div style="margin-top:2rem;">
                <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:2px; color:#333; text-transform:uppercase; margin-bottom:1rem;">
                    {len(lines)} sentence{'s' if len(lines)>1 else ''} analyzed
                </div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)
