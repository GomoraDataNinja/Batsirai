import streamlit as st
import pandas as pd
import re
import html
from googleapiclient.discovery import build
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import numpy as np
import time
import os
import hashlib
import warnings

warnings.filterwarnings("ignore")

APP_VERSION = "3.3.0"
APP_NAME = "Batsirai"
DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "production")
SESSION_TIMEOUT_MINUTES = 60

st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def get_org_password():
    env_pw = os.environ.get("APP_PASSWORD", "").strip()
    if env_pw:
        return env_pw
    try:
        sec_pw = str(st.secrets.get("app_password", "")).strip()
        if sec_pw:
            return sec_pw
    except Exception:
        pass
    return "youtube2024"

ORG_PASSWORD = get_org_password()

THEME = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "panel2": "#f7f7f7",
    "text": "#111111",
    "muted": "#5b5b5b",
    "border": "rgba(0,0,0,0.10)",
    "border2": "rgba(0,0,0,0.14)",
    "accent": "#d71e28",
    "accent2": "#b5161f",
    "good": "#168a45",
    "bad": "#d11a2a",
    "neutral": "#6b7280",
}

SENTIMENT_COLORS = {
    "Positive": THEME["good"],
    "Neutral": THEME["neutral"],
    "Negative": THEME["bad"],
}

def apply_style():
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {THEME['bg']};
            --panel: {THEME['panel']};
            --panel2: {THEME['panel2']};
            --text: {THEME['text']};
            --muted: {THEME['muted']};
            --border: {THEME['border']};
            --border2: {THEME['border2']};
            --accent: {THEME['accent']};
            --accent2: {THEME['accent2']};
            --good: {THEME['good']};
            --bad: {THEME['bad']};
            --neutral: {THEME['neutral']};
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: var(--bg) !important;
            color: var(--text) !important;
        }}

        [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }}

        .block-container {{
            max-width: 1120px;
            padding-top: 2.6rem !important;
            padding-bottom: 2.2rem !important;
        }}

        html, body, .stApp, .stMarkdown, .stText, p, span, div, label {{
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif !important;
            color: var(--text) !important;
        }}

        section[data-testid="stSidebar"] {{
            background: #ffffff !important;
            border-right: 1px solid var(--border) !important;
        }}

        .card {{
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            padding: 18px 18px !important;
        }}

        .card-soft {{
            background: var(--panel2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            padding: 18px 18px !important;
        }}

        .hero {{
            border: 1px solid var(--border) !important;
            border-radius: 22px !important;
            padding: 26px 22px !important;
            background:
                radial-gradient(900px 260px at 50% -10%, rgba(215,30,40,0.10), transparent 60%),
                linear-gradient(180deg, #ffffff, #ffffff) !important;
        }}

        .title {{
            font-size: 30px !important;
            font-weight: 800 !important;
            letter-spacing: 0.2px !important;
            margin: 0 !important;
        }}

        .subtitle {{
            margin-top: 8px !important;
            color: var(--muted) !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }}

        .chip {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 8px !important;
            padding: 6px 12px !important;
            border-radius: 999px !important;
            border: 1px solid var(--border) !important;
            background: #ffffff !important;
            font-size: 12px !important;
            font-weight: 650 !important;
            color: var(--muted) !important;
        }}

        .chip-dot {{
            width: 8px !important;
            height: 8px !important;
            border-radius: 999px !important;
            display: inline-block !important;
            background: var(--accent) !important;
        }}

        .metric {{
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            padding: 14px 14px !important;
            background: #ffffff !important;
        }}

        .metric-k {{
            font-size: 12px !important;
            color: var(--muted) !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.9px !important;
        }}

        .metric-v {{
            font-size: 26px !important;
            font-weight: 850 !important;
            margin-top: 6px !important;
        }}

        .muted {{
            color: var(--muted) !important;
        }}

        .card h1, .card h2, .card h3, .card h4,
        .card-soft h1, .card-soft h2, .card-soft h3, .card-soft h4 {{
            font-weight: 900 !important;
        }}
        .card strong, .card b, .card-soft strong, .card-soft b {{
            font-weight: 900 !important;
        }}
        .card div[style*="font-weight:800"],
        .card div[style*="font-weight: 800"],
        .card-soft div[style*="font-weight:800"],
        .card-soft div[style*="font-weight: 800"] {{
            font-weight: 900 !important;
        }}

        div.stButton > button,
        button,
        button[kind="primary"],
        button[kind="secondary"],
        [data-testid="baseButton-primary"] > button,
        [data-testid="baseButton-secondary"] > button {{
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 14px !important;
            padding: 0.7rem 1rem !important;
            font-weight: 750 !important;
            color: #ffffff !important;
        }}

        div.stButton > button:hover,
        button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover,
        [data-testid="baseButton-primary"] > button:hover,
        [data-testid="baseButton-secondary"] > button:hover {{
            background: var(--accent2) !important;
            border: 1px solid var(--accent2) !important;
        }}

        div[data-baseweb="base-input"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: #ffffff !important;
            border: 1px solid var(--border2) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="base-input"] input,
        div[data-baseweb="input"] input {{
            background: transparent !important;
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }}

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {{
            color: var(--text) !important;
            fill: var(--text) !important;
        }}

        div[data-baseweb="popover"] {{
            background: #ffffff !important;
            border: 1px solid var(--border2) !important;
            box-shadow: 0 12px 28px rgba(0,0,0,0.10) !important;
        }}

        /* Fix dropdown black background on deploy (menus / listboxes rendered in portal) */
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] ul,
        div[data-baseweb="menu"] li,
        ul[role="listbox"],
        ul[role="listbox"] li,
        [role="option"] {{
            background: #ffffff !important;
            color: var(--text) !important;
        }}

        div[data-baseweb="menu"] li:hover,
        ul[role="listbox"] li:hover,
        [role="option"]:hover {{
            background: rgba(215,30,40,0.08) !important;
        }}

        div[data-baseweb="menu"] li[aria-selected="true"],
        ul[role="listbox"] li[aria-selected="true"],
        [role="option"][aria-selected="true"] {{
            background: rgba(215,30,40,0.12) !important;
        }}

        div[data-baseweb="menu"] * {{
            color: var(--text) !important;
        }}

        span[data-baseweb="tag"] {{
            background: rgba(215,30,40,0.12) !important;
            border: 1px solid rgba(215,30,40,0.25) !important;
            color: var(--text) !important;
        }}
        span[data-baseweb="tag"] svg {{
            fill: var(--text) !important;
        }}

        .stTabs [data-baseweb="tab-list"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            gap: 14px !important;
            width: 100% !important;
            flex-wrap: wrap !important;
            margin-top: 10px !important;
            padding: 0 6px !important;
        }}

        .stTabs [data-baseweb="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] {{
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            margin-right: 0 !important;
            padding: 14px 18px !important;
            font-weight: 850 !important;
            font-size: 15px !important;
            min-width: 150px !important;
            text-align: center !important;
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
            background: rgba(215,30,40,0.10) !important;
            border: 1px solid rgba(215,30,40,0.35) !important;
        }}

        [data-testid="stDataFrame"] {{
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
        }}

        /* Best pick link styling */
        a.batsirai-link {{
            color: var(--accent) !important;
            text-decoration: underline !important;
            font-weight: 750 !important;
        }}
        a.batsirai-link:hover {{
            color: var(--accent2) !important;
        }}

        @media (max-width: 820px) {{
            .stTabs [data-baseweb="tab"],
            div[data-testid="stTabs"] [data-baseweb="tab"] {{
                min-width: 0 !important;
                padding: 12px 14px !important;
                font-size: 14px !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_style()

def touch():
    st.session_state.last_activity = datetime.now()

def is_timed_out():
    last = st.session_state.get("last_activity")
    if not last:
        return False
    return (datetime.now() - last).total_seconds() > SESSION_TIMEOUT_MINUTES * 60

def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    safe_rerun()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()
if "topic_results" not in st.session_state:
    st.session_state.topic_results = None
if "topic_query" not in st.session_state:
    st.session_state.topic_query = ""

def login_screen():
    st.markdown('<div style="height: 1.8rem;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.25, 1])
    with c2:
        st.markdown(
            f"""
            <div class="card" style="margin-top: 10vh;">
                <div class="title" style="text-align:center;">{APP_NAME}</div>
                <div class="subtitle" style="text-align:center;">
                    Sign in to continue.
                </div>
                <div style="height: 14px;"></div>
                <div style="display:flex; justify-content:center;">
                    <div class="chip"><span class="chip-dot"></span> Version {APP_VERSION} • {DEPLOYMENT_MODE.title()}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=True):
            pw = st.text_input("Password", type="password", placeholder="Organisation password")
            ok = st.form_submit_button("Sign in", use_container_width=True)

        if ok:
            if pw == ORG_PASSWORD:
                st.session_state.authenticated = True
                touch()
                safe_rerun()
            else:
                st.error("Wrong password.")

if st.session_state.authenticated and is_timed_out():
    st.session_state.authenticated = False
    st.warning("Session timed out. Sign in again.")
    login_screen()
    st.stop()

if not st.session_state.authenticated:
    login_screen()
    st.stop()

touch()

def youtube_client():
    try:
        api_key = st.secrets["youtube_api_key"]
    except Exception:
        api_key = None

    if not api_key or not str(api_key).strip():
        st.error("Missing YouTube API key. Add youtube_api_key in Streamlit secrets.")
        return None

    return build("youtube", "v3", developerKey=str(api_key).strip())

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can did do does doing down during each few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom why will with you your
yours yourself yourselves
""".split())

def shorten(text, n=160):
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "..."

def clean_tokens(text: str):
    text = str(text or "").lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [p.strip() for p in text.split() if p.strip()]
    parts = [p for p in parts if p not in STOPWORDS and len(p) > 2]
    return parts

def jaccard_similarity(a_tokens, b_tokens):
    a = set(a_tokens)
    b = set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / max(1, len(a.union(b)))

def minmax(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn = float(s.min())
    mx = float(s.max())
    if mx - mn < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def parse_published_at(s):
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.NaT

def get_video_comments(youtube, video_id, max_comments=150):
    all_comments = []
    next_page_token = None

    while len(all_comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()

            for item in response.get("items", []):
                s = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append(
                    {
                        "video_id": video_id,
                        "comment": s.get("textDisplay", ""),
                        "published_at": s.get("publishedAt", ""),
                        "like_count": s.get("likeCount", 0),
                        "author": s.get("authorDisplayName", "Unknown"),
                    }
                )
                if len(all_comments) >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        except Exception as e:
            msg = str(e)
            if "commentsDisabled" in msg:
                return []
            return []

    return all_comments

def analyze_sentiment_comment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return score, label

def search_videos_by_topic(youtube, topic, max_videos=30, order="relevance", published_after=None, region_code=None):
    results = []
    next_page_token = None

    while len(results) < max_videos:
        try:
            req = youtube.search().list(
                part="snippet",
                q=topic,
                type="video",
                maxResults=min(50, max_videos - len(results)),
                pageToken=next_page_token,
                order=order,
                safeSearch="none",
                publishedAfter=published_after if published_after else None,
                regionCode=region_code if region_code else None,
            )
            resp = req.execute()
            for item in resp.get("items", []):
                vid = item.get("id", {}).get("videoId")
                sn = item.get("snippet", {})
                if not vid:
                    continue
                results.append(
                    {
                        "video_id": vid,
                        "title": sn.get("title", ""),
                        "description": sn.get("description", ""),
                        "channel": sn.get("channelTitle", ""),
                        "published_at": sn.get("publishedAt", ""),
                    }
                )
                if len(results) >= max_videos:
                    break

            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        except Exception:
            break

    return results

def fetch_video_stats(youtube, video_ids):
    out = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        try:
            resp = youtube.videos().list(part="snippet,statistics", id=",".join(chunk)).execute()
            for item in resp.get("items", []):
                vid = item.get("id")
                stats = item.get("statistics", {}) or {}
                sn = item.get("snippet", {}) or {}
                out[vid] = {
                    "views": int(stats.get("viewCount", 0) or 0),
                    "likes": int(stats.get("likeCount", 0) or 0),
                    "comment_count": int(stats.get("commentCount", 0) or 0),
                    "title": sn.get("title", ""),
                    "channel": sn.get("channelTitle", ""),
                    "published_at": sn.get("publishedAt", ""),
                    "description": sn.get("description", ""),
                }
        except Exception:
            continue
    return out

def donut_chart(sentiment_counts, title, center_text):
    labels = list(sentiment_counts.index)
    values = list(sentiment_counts.values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=[SENTIMENT_COLORS.get(x, THEME["neutral"]) for x in labels]),
                textinfo="percent",
                textfont=dict(color=THEME["text"]),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, color=THEME["text"])),
        showlegend=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=THEME["text"]),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="left",
            x=0,
            font=dict(size=13, color=THEME["text"]),
            bgcolor="rgba(255,255,255,1)",
            bordercolor=THEME["border2"],
            borderwidth=1,
        ),
        annotations=[
            dict(
                text=center_text,
                x=0.5,
                y=0.5,
                font=dict(size=14, color=THEME["text"]),
                showarrow=False,
            )
        ],
    )
    return fig

def build_reasons(row):
    reasons = []

    if row.get("relevance_score", 0) >= 0.55:
        reasons.append("Strong match to your topic based on title and comment language.")
    elif row.get("relevance_score", 0) >= 0.35:
        reasons.append("Decent topic match, still aligned with what you searched.")

    if row.get("engagement_rate", 0) >= row.get("engagement_rate_p75", 0):
        reasons.append("High engagement for its views, people interact a lot here.")

    if row.get("pos_pct", 0) >= 55 and row.get("neg_pct", 0) <= 20:
        reasons.append("Audience reaction leans positive with low negativity.")
    elif row.get("neg_pct", 0) >= 35 and row.get("engagement_rate", 0) > row.get("engagement_rate_p50", 0):
        reasons.append("Controversial but popular, lots of debate in the comments.")

    if row.get("freshness_days", 9999) <= 30:
        reasons.append("Fresh upload, the discussion is current.")

    kw = str(row.get("top_keywords", "")).strip()
    if kw:
        reasons.append(f"Common terms in comments: {kw}.")

    if not reasons:
        reasons.append("Balanced performance across engagement, sentiment, and topic match.")

    return reasons[:4]

def extract_phrases_from_text(text: str):
    t = str(text or "").strip()
    if not t:
        return []
    try:
        phrases = [str(p).lower().strip() for p in TextBlob(t).noun_phrases]
    except Exception:
        phrases = []
    cleaned = []
    for p in phrases:
        p = re.sub(r"[^a-z0-9\s]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        if len(p) < 3:
            continue
        if len(p.split()) > 4:
            continue
        if p in STOPWORDS:
            continue
        cleaned.append(p)
    return cleaned

def build_theme_points(dfc: pd.DataFrame, label: str = None, top_k: int = 4):
    if dfc is None or dfc.empty:
        return []

    df = dfc.copy()
    if label:
        df = df[df["sentiment"] == label].copy()
        if df.empty:
            return []

    phrase_counts = {}
    phrase_examples = {}
    total = int(len(df))

    for c in df["comment"].astype(str).tolist():
        phrases = extract_phrases_from_text(c)
        if not phrases:
            continue
        for ph in set(phrases):
            phrase_counts[ph] = phrase_counts.get(ph, 0) + 1
            if ph not in phrase_examples:
                phrase_examples[ph] = c

    if not phrase_counts:
        return []

    items = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[: max(10, top_k * 3)]

    points = []
    used = set()
    for ph, cnt in items:
        if ph in used:
            continue
        used.add(ph)

        pct = (cnt / max(1, total)) * 100.0
        ex = shorten(phrase_examples.get(ph, ""), 150)

        if label == "Positive":
            sentence = f"People like how it covers {ph}."
        elif label == "Negative":
            sentence = f"Some viewers complain about {ph}."
        else:
            sentence = f"People keep mentioning {ph}."

        points.append({"sentence": sentence, "pct": pct, "count": cnt, "example": ex})

        if len(points) >= top_k:
            break

    return points

def normalize_question(text: str):
    s = str(text or "").strip().lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9\?\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s.endswith("?") and "?" in s:
        s = s.replace("?", "").strip() + "?"
    return s

def build_question_points(dfc: pd.DataFrame, top_k: int = 5):
    if dfc is None or dfc.empty:
        return []

    df = dfc.copy()
    df["comment_str"] = df["comment"].astype(str)
    qdf = df[df["comment_str"].str.contains(r"\?", regex=True, na=False)].copy()
    if qdf.empty:
        return []

    total_all = int(len(df))
    counts = {}
    example = {}
    for txt in qdf["comment_str"].tolist():
        raw = txt.strip()
        if not raw:
            continue
        nrm = normalize_question(raw)
        if len(nrm) < 6:
            continue
        if len(nrm) > 160:
            nrm = nrm[:160].rstrip() + "?"
        counts[nrm] = counts.get(nrm, 0) + 1
        if nrm not in example:
            example[nrm] = raw

    if not counts:
        return []

    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    out = []
    for q, cnt in items:
        pct = (cnt / max(1, total_all)) * 100.0
        ex = shorten(example.get(q, q), 150)
        out.append({"sentence": "People keep asking: " + q, "pct": pct, "count": cnt, "example": ex})
        if len(out) >= top_k:
            break

    return out

def build_final_answer(res: dict):
    dv = res.get("videos_df")
    dc = res.get("comments_df")
    topic = res.get("topic", "")

    if dv is None or dv.empty:
        return {
            "headline": "No results yet",
            "bullets": ["Run a topic search first."],
            "overall": None,
            "top_pick": None,
            "themes": {"common": [], "positive": [], "negative": [], "questions": []},
        }

    scanned = int(len(dv))
    top_pick = dv.iloc[0].to_dict()
    top_title = str(top_pick.get("title", "")).strip()
    top_channel = str(top_pick.get("channel", "")).strip()
    top_url = str(top_pick.get("video_url", "")).strip()

    bullets = []
    overall = None

    if dc is None or dc.empty:
        bullets.append(f"I scanned {scanned} videos for this topic.")
        bullets.append("I could not sample comments for this run, so ranking used engagement, topic match, and recency.")
        bullets.append(f"Best pick right now: {top_title} by {top_channel}.")
        bullets.append("Try increasing Comments per video, or pick a different time window or region.")
        overall = {"sentiment_counts": pd.Series({"Positive": 0, "Neutral": 0, "Negative": 0})}
        return {
            "headline": f"What this topic looks like on YouTube: {topic}",
            "bullets": bullets,
            "overall": overall,
            "top_pick": {"title": top_title, "channel": top_channel, "url": top_url},
            "themes": {"common": [], "positive": [], "negative": [], "questions": []},
        }

    sc = dc["sentiment"].value_counts()
    pos = int(sc.get("Positive", 0))
    neu = int(sc.get("Neutral", 0))
    neg = int(sc.get("Negative", 0))
    total = max(1, int(len(dc)))

    pos_pct = (pos / total) * 100
    neg_pct = (neg / total) * 100

    vibe = "mixed"
    if pos_pct >= 55 and neg_pct <= 20:
        vibe = "mostly positive"
    elif neg_pct >= 35 and pos_pct <= 40:
        vibe = "mostly negative"
    elif neg_pct >= 30 and pos_pct >= 45:
        vibe = "split"

    bullets.append(f"I scanned {scanned} videos for this topic.")
    bullets.append(f"Comment vibe looks {vibe}: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative.")
    bullets.append(f"Best pick right now: {top_title} by {top_channel}.")
    bullets.append("It ranks high because it combines engagement, positive reaction, and topic match better than the rest.")

    common_points = build_theme_points(dc, label=None, top_k=4)
    pos_points = build_theme_points(dc, label="Positive", top_k=3)
    neg_points = build_theme_points(dc, label="Negative", top_k=3)
    q_points = build_question_points(dc, top_k=5)

    overall = {"sentiment_counts": pd.Series({"Positive": pos, "Neutral": neu, "Negative": neg})}

    return {
        "headline": f"What this topic looks like on YouTube: {topic}",
        "bullets": bullets,
        "overall": overall,
        "top_pick": {"title": top_title, "channel": top_channel, "url": top_url},
        "themes": {"common": common_points, "positive": pos_points, "negative": neg_points, "questions": q_points},
    }

def run_topic_analysis(topic, max_videos, comments_per_video, order, time_window_days, region_code):
    yt = youtube_client()
    if yt is None:
        return None

    topic = (topic or "").strip()
    if not topic:
        return None

    published_after = None
    if time_window_days and int(time_window_days) > 0:
        dt = datetime.now(timezone.utc) - pd.Timedelta(days=int(time_window_days))
        published_after = dt.isoformat().replace("+00:00", "Z")

    videos = search_videos_by_topic(
        yt,
        topic=topic,
        max_videos=max_videos,
        order=order,
        published_after=published_after,
        region_code=region_code,
    )
    if not videos:
        return None

    ids = [v["video_id"] for v in videos]
    stats = fetch_video_stats(yt, ids)

    video_rows = []
    comment_rows = []

    progress = st.progress(0)
    status = st.empty()

    topic_tokens = clean_tokens(topic)

    for idx, vid in enumerate(ids, start=1):
        meta = stats.get(vid, {})
        title = meta.get("title") or next((x["title"] for x in videos if x["video_id"] == vid), "")
        desc = meta.get("description") or next((x["description"] for x in videos if x["video_id"] == vid), "")
        channel = meta.get("channel") or next((x["channel"] for x in videos if x["video_id"] == vid), "")
        published_at = meta.get("published_at") or next((x["published_at"] for x in videos if x["video_id"] == vid), "")

        status.markdown(f"<div class='muted'>Scanning video {idx} of {len(ids)}...</div>", unsafe_allow_html=True)

        comments = get_video_comments(yt, vid, max_comments=comments_per_video)

        dfc = pd.DataFrame(comments) if comments else pd.DataFrame(columns=["video_id", "comment", "published_at", "like_count", "author"])
        if not dfc.empty:
            dfc["published_at"] = pd.to_datetime(dfc["published_at"], errors="coerce", utc=True)
            scores = []
            labels = []
            for txt in dfc["comment"].astype(str).tolist():
                sc0, lb0 = analyze_sentiment_comment(txt)
                scores.append(sc0)
                labels.append(lb0)
            dfc["sentiment_score"] = scores
            dfc["sentiment"] = labels
            comment_rows.extend(dfc.to_dict("records"))

        n_comments_sampled = int(len(dfc)) if dfc is not None else 0
        avg_sent = float(np.mean(dfc["sentiment_score"])) if n_comments_sampled > 0 else 0.0
        pos_pct = float((dfc["sentiment"] == "Positive").mean() * 100) if n_comments_sampled > 0 else 0.0
        neg_pct = float((dfc["sentiment"] == "Negative").mean() * 100) if n_comments_sampled > 0 else 0.0
        neu_pct = float((dfc["sentiment"] == "Neutral").mean() * 100) if n_comments_sampled > 0 else 0.0

        views = int(meta.get("views", 0) or 0)
        likes = int(meta.get("likes", 0) or 0)
        comment_count = int(meta.get("comment_count", 0) or 0)

        engagement_rate = (likes + comment_count) / max(1, views)

        pub_dt = parse_published_at(published_at)
        freshness_days = 9999
        if pd.notna(pub_dt):
            freshness_days = int((datetime.now(timezone.utc) - pub_dt.to_pydatetime()).total_seconds() / 86400)

        text_tokens = clean_tokens(title + " " + desc)
        relevance = jaccard_similarity(topic_tokens, text_tokens)

        top_keywords = ""
        if n_comments_sampled > 0:
            all_tokens = []
            for t0 in dfc["comment"].astype(str).tolist()[: min(300, n_comments_sampled)]:
                all_tokens.extend(clean_tokens(t0))
            if all_tokens:
                vc = pd.Series(all_tokens).value_counts().head(5)
                top_keywords = ", ".join(vc.index.tolist())

        video_rows.append(
            {
                "video_id": vid,
                "title": title,
                "channel": channel,
                "published_at": published_at,
                "views": views,
                "likes": likes,
                "comment_count": comment_count,
                "sampled_comments": n_comments_sampled,
                "engagement_rate": float(engagement_rate),
                "avg_sentiment": float(avg_sent),
                "pos_pct": float(pos_pct),
                "neu_pct": float(neu_pct),
                "neg_pct": float(neg_pct),
                "relevance_score": float(relevance),
                "freshness_days": int(freshness_days),
                "top_keywords": top_keywords,
                "video_url": f"https://www.youtube.com/watch?v={vid}",
            }
        )

        progress.progress(int((idx / len(ids)) * 100))

    status.empty()
    progress.empty()

    if not video_rows:
        return None

    dv = pd.DataFrame(video_rows)

    dv["freshness_score"] = 1.0 / (1.0 + (pd.to_numeric(dv["freshness_days"], errors="coerce").fillna(9999) / 30.0))
    dv["sentiment_score_video"] = (dv["avg_sentiment"].clip(-1, 1) + 1) / 2.0
    dv["quality_score"] = (minmax(dv["sampled_comments"]) * 0.6 + minmax(dv["comment_count"]) * 0.4).clip(0, 1)

    dv["engagement_norm"] = minmax(dv["engagement_rate"])
    dv["relevance_norm"] = minmax(dv["relevance_score"])
    dv["freshness_norm"] = minmax(dv["freshness_score"])
    dv["sentiment_norm"] = minmax(dv["sentiment_score_video"])

    dv["final_score"] = (
        0.40 * dv["engagement_norm"]
        + 0.30 * dv["sentiment_norm"]
        + 0.20 * dv["relevance_norm"]
        + 0.10 * dv["freshness_norm"]
    ) * (0.70 + 0.30 * dv["quality_score"])

    dv = dv.sort_values("final_score", ascending=False).reset_index(drop=True)
    dv["rank"] = np.arange(1, len(dv) + 1)

    dv["engagement_rate_p50"] = float(dv["engagement_rate"].median())
    dv["engagement_rate_p75"] = float(dv["engagement_rate"].quantile(0.75))

    reasons_col = []
    for _, r in dv.iterrows():
        reasons = build_reasons(r.to_dict())
        reasons_col.append("\n".join([f"- {x}" for x in reasons]))
    dv["why_this_ranked"] = reasons_col

    res = {
        "topic": topic,
        "videos_df": dv,
        "comments_df": pd.DataFrame(comment_rows) if comment_rows else pd.DataFrame(columns=["video_id", "comment", "published_at", "like_count", "author", "sentiment_score", "sentiment"]),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    res["final_answer"] = build_final_answer(res)
    return res

st.markdown(
    f"""
    <div class="hero" style="text-align:center;">
        <div class="title">{APP_NAME}</div>
        <div class="subtitle">Welcome to Batsirai. Search a topic and I will find videos, read comments, then rank the best picks.</div>
        <div style="height: 12px;"></div>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <div class="chip"><span class="chip-dot"></span> Secure session</div>
            <div class="chip">Session {st.session_state.session_id}</div>
            <div class="chip">Mode {DEPLOYMENT_MODE.title()}</div>
            <div class="chip">Version {APP_VERSION}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

tabs = st.tabs(["Search", "Top 10", "Explore", "Export"])

with tabs[0]:
    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Topic search</div>
            <div class="subtitle">Type what you want to know. I search YouTube and run the analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    c1, c2 = st.columns([3, 1])
    with c1:
        topic = st.text_input(
            "Topic",
            value=st.session_state.topic_query,
            placeholder="Example: best excel tutor, study tips, how to budget",
        )
    with c2:
        order = st.selectbox("Sort results by", options=["relevance", "date", "viewCount", "rating"], index=0)

    r1, r2, r3, r4 = st.columns([1.1, 1.1, 1.1, 1.2])
    with r1:
        max_videos = st.slider("Videos to scan", 10, 80, 30, step=5)
    with r2:
        comments_per_video = st.slider("Comments per video", 50, 300, 150, step=25)
    with r3:
        time_window_days = st.selectbox(
            "Time window",
            options=[0, 7, 30, 90, 365],
            index=2,
            format_func=lambda x: "Any time" if x == 0 else f"Last {x} days",
        )
    with r4:
        region_code = st.text_input("Region code (optional)", value="", placeholder="Example: ZW, ZA, GB")

    b1, b2 = st.columns([1, 5])
    with b1:
        run = st.button("Search and analyze", use_container_width=True)
    with b2:
        if st.button("Clear results", use_container_width=True):
            st.session_state.topic_results = None
            st.session_state.topic_query = ""
            safe_rerun()

    if run:
        topic_clean = (topic or "").strip()
        if not topic_clean:
            st.error("Type a topic first.")
        else:
            st.session_state.topic_query = topic_clean
            with st.spinner("Searching and analyzing videos..."):
                res = run_topic_analysis(
                    topic=topic_clean,
                    max_videos=int(max_videos),
                    comments_per_video=int(comments_per_video),
                    order=order,
                    time_window_days=int(time_window_days),
                    region_code=(region_code or "").strip() or None,
                )
            if res is None:
                st.error("No results. Try a different topic or check your API key and quota.")
            else:
                st.session_state.topic_results = res
                st.success("Done. Check Top 10.")
                safe_rerun()

with tabs[1]:
    res = st.session_state.topic_results
    if not res:
        st.markdown(
            """
            <div class="card-soft">
                <div style="font-size:16px; font-weight:800;">No results yet</div>
                <div class="subtitle">Use the Search tab to analyze a topic.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    dv = res["videos_df"].copy()
    topic = res["topic"]
    top10 = dv.head(10).copy()

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Top 10 videos for: {html.escape(topic)}</div>
            <div class="subtitle">Ranking uses engagement, sentiment, topic match, and recency.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    total_scanned = len(dv)
    avg_sent = float(dv["avg_sentiment"].mean()) if total_scanned else 0.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric'><div class='metric-k'>Scanned</div><div class='metric-v'>{total_scanned}</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric'><div class='metric-k'>Avg sentiment</div><div class='metric-v'>{avg_sent:.2f}</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric'><div class='metric-k'>Generated</div><div class='metric-v'>{res['generated_at']}</div></div>", unsafe_allow_html=True)

    st.markdown("")

    fa = res.get("final_answer", {})
    headline = fa.get("headline", "")
    bullets = fa.get("bullets", [])
    overall = fa.get("overall", None)
    top_pick = fa.get("top_pick", None)
    themes = fa.get("themes", {"common": [], "positive": [], "negative": [], "questions": []})

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Bot answer</div>
            <div class="subtitle">{html.escape(headline)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            "<div class='card-soft' style='white-space:pre-wrap; line-height:1.65; font-size:13px;'>"
            + "\n".join([f"- {html.escape(b)}" for b in bullets])
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown(
            """
            <div class="card">
                <div style="font-size:16px; font-weight:800;">What people are actually saying</div>
                <div class="subtitle">Each point includes an example comment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

        def render_points(title, points):
            if not points:
                return
            st.markdown(
                f"""
                <div class="card-soft" style="margin-top: 10px;">
                    <div style="font-weight:800; font-size:14px;">{html.escape(title)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for p in points:
                sentence = html.escape(p.get("sentence", ""))
                pct = float(p.get("pct", 0.0))
                ex = html.escape(p.get("example", ""))
                st.markdown(
                    f"""
                    <div class="card-soft" style="margin-top: 10px;">
                        <div style="font-weight:800; font-size:13px;">- {sentence}</div>
                        <div class="muted" style="margin-top:6px; font-size:12px;">Shows up in about {pct:.1f}% of sampled comments.</div>
                        <div style="margin-top:8px; font-size:12.5px; line-height:1.55;">Example: "{ex}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        render_points("People keep mentioning", themes.get("common", []))
        render_points("People like", themes.get("positive", []))
        render_points("People complain about", themes.get("negative", []))
        render_points("People keep asking", themes.get("questions", []))

        if top_pick and str(top_pick.get("url", "")).strip():
            tp_title = html.escape(top_pick.get("title", ""))
            tp_channel = html.escape(top_pick.get("channel", ""))
            tp_url = html.escape(top_pick.get("url", ""))
            st.markdown(
                f"""
                <div class="card-soft" style="margin-top: 12px;">
                    <div style="font-weight:800;">Best pick right now</div>
                    <div class="subtitle" style="margin-top:6px;">{tp_title}</div>
                    <div class="muted" style="font-size:13px; margin-top:4px;">{tp_channel}</div>
                    <div class="muted" style="font-size:12px; margin-top:10px;">
                        <a class="batsirai-link" href="{tp_url}" target="_blank" rel="noopener noreferrer">{tp_url}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        if overall and isinstance(overall, dict):
            sc = overall.get("sentiment_counts", None)
            if sc is not None and isinstance(sc, pd.Series) and sc.sum() > 0:
                dom = sc.idxmax()
                dom_pct = (float(sc.max()) / max(1.0, float(sc.sum()))) * 100.0
                center = f"{dom}<br>{dom_pct:.0f}%"
                fig = donut_chart(sc, "Overall comment sentiment", center)
                st.plotly_chart(fig, use_container_width=True, key="top10_overall_donut")
            else:
                st.markdown(
                    """
                    <div class="card-soft">
                        <div style="font-weight:800;">Sentiment chart</div>
                        <div class="subtitle">No sampled comments to plot for this run.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Top 10 table</div>
            <div class="subtitle">Export from the Export tab if you want files.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    show_cols = [
        "rank",
        "title",
        "channel",
        "views",
        "likes",
        "comment_count",
        "pos_pct",
        "neg_pct",
        "relevance_score",
        "final_score",
        "video_url",
    ]
    t = top10[show_cols].copy()
    t["views"] = t["views"].map(lambda x: f"{int(x):,}")
    t["likes"] = t["likes"].map(lambda x: f"{int(x):,}")
    t["comment_count"] = t["comment_count"].map(lambda x: f"{int(x):,}")
    t["pos_pct"] = t["pos_pct"].map(lambda x: f"{x:.1f}%")
    t["neg_pct"] = t["neg_pct"].map(lambda x: f"{x:.1f}%")
    t["relevance_score"] = t["relevance_score"].map(lambda x: f"{x:.2f}")
    t["final_score"] = t["final_score"].map(lambda x: f"{x:.3f}")

    st.dataframe(t, use_container_width=True, height=340)

    st.markdown("")
    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Recommendations</div>
            <div class="subtitle">Each video includes a short reason for its rank.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    for _, row in top10.iterrows():
        reasons = row["why_this_ranked"]
        title = html.escape(str(row["title"]))
        channel = html.escape(str(row["channel"]))
        url = html.escape(str(row["video_url"]))
        chip = f"<span class='chip'><span class='chip-dot'></span> Rank {int(row['rank'])}</span>"

        st.markdown(
            f"""
            <div class="card-soft" style="margin-top: 10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                    <div style="font-weight:800; font-size:15px;">{title}</div>
                    <div>{chip}</div>
                </div>
                <div class="muted" style="margin-top:6px; font-size:13px;">{channel}</div>
                <div style="margin-top:10px; line-height:1.6; font-size:13px; white-space:pre-wrap;">{html.escape(str(reasons))}</div>
                <div class="muted" style="margin-top:10px; font-size:12px;">{url}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[2]:
    res = st.session_state.topic_results
    if not res:
        st.markdown(
            """
            <div class="card-soft">
                <div style="font-size:16px; font-weight:800;">No results yet</div>
                <div class="subtitle">Use the Search tab to analyze a topic.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    dv = res["videos_df"].copy()
    dc = res["comments_df"].copy()

    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Explore</div>
            <div class="subtitle">Inspect one video and its sampled comments.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    left, right = st.columns([1.1, 1])

    with left:
        dv2 = dv.head(30).copy()
        dv2["title_short"] = dv2["title"].astype(str).apply(lambda x: (x[:55] + "...") if len(x) > 55 else x)

        fig = px.bar(
            dv2.sort_values("final_score", ascending=True),
            x="final_score",
            y="title_short",
            orientation="h",
            title="Top scores (first 30)",
        )
        fig.update_layout(
            height=520,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color=THEME["text"], size=13),
            title_font=dict(color=THEME["text"], size=16),
            margin=dict(l=10, r=10, t=60, b=20),
        )
        fig.update_xaxes(title="Score", showgrid=True, tickfont=dict(size=12, color=THEME["text"]))
        fig.update_yaxes(title="Video", showgrid=False, tickfont=dict(size=12, color=THEME["text"]))

        st.plotly_chart(fig, use_container_width=True, key="explore_top_scores_bar")

    with right:
        if not dc.empty:
            s_counts = dc["sentiment"].value_counts()
            dom = s_counts.idxmax()
            dom_pct = (s_counts.max() / len(dc)) * 100
            center = f"{dom}<br>{dom_pct:.0f}%"
            fig2 = donut_chart(s_counts, "Overall comment sentiment", center)
            st.plotly_chart(fig2, use_container_width=True, key="explore_overall_donut")
        else:
            st.markdown(
                """
                <div class="card-soft">
                    <div style="font-size:16px; font-weight:800;">No comments sampled</div>
                    <div class="subtitle">Some videos may block comments or restrict access.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    pick = st.selectbox(
        "Video",
        options=dv["video_id"].tolist(),
        format_func=lambda x: dv.loc[dv["video_id"] == x, "title"].values[0][:80],
        key="explore_video_pick",
    )

    row = dv[dv["video_id"] == pick].iloc[0].to_dict()
    v_url = row.get("video_url", "")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='metric'><div class='metric-k'>Views</div><div class='metric-v'>{int(row.get('views',0)):,}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric'><div class='metric-k'>Likes</div><div class='metric-v'>{int(row.get('likes',0)):,}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric'><div class='metric-k'>Comments</div><div class='metric-v'>{int(row.get('comment_count',0)):,}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric'><div class='metric-k'>Rank</div><div class='metric-v'>{int(row.get('rank',0))}</div></div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        f"""
        <div class='muted' style='font-size:12px;'>
            Link:
            <a class="batsirai-link" href="{html.escape(str(v_url))}" target="_blank" rel="noopener noreferrer">{html.escape(str(v_url))}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    if dc.empty:
        st.info("No comment data available for this run.")
    else:
        dcv = dc[dc["video_id"] == pick].copy()
        if dcv.empty:
            st.info("No sampled comments for this video.")
        else:
            f1, f2, f3 = st.columns([1.2, 1.2, 1])
            with f1:
                sentiment_filter = st.multiselect(
                    "Filter sentiment",
                    options=["Positive", "Neutral", "Negative"],
                    default=["Positive", "Neutral", "Negative"],
                    key="explore_sent_filter",
                )
            with f2:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Newest", "Oldest", "Most Likes", "Highest Sentiment", "Lowest Sentiment"],
                    key="explore_sort_by",
                )
            with f3:
                n_show = st.slider("Rows", 10, 80, 20, key="explore_rows")

            dcv = dcv[dcv["sentiment"].isin(sentiment_filter)]
            mapping = {
                "Newest": ("published_at", False),
                "Oldest": ("published_at", True),
                "Most Likes": ("like_count", False),
                "Highest Sentiment": ("sentiment_score", False),
                "Lowest Sentiment": ("sentiment_score", True),
            }
            col, asc = mapping[sort_by]
            dcv = dcv.sort_values(col, ascending=asc).head(n_show).copy()
            dcv["published_at"] = pd.to_datetime(dcv["published_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M")

            st.dataframe(
                dcv[["published_at", "author", "like_count", "sentiment", "sentiment_score", "comment"]],
                use_container_width=True,
                height=520,
            )

with tabs[3]:
    res = st.session_state.topic_results
    if not res:
        st.markdown(
            """
            <div class="card-soft">
                <div style="font-size:16px; font-weight:800;">No results yet</div>
                <div class="subtitle">Use the Search tab to analyze a topic.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    dv = res["videos_df"].copy()
    dc = res["comments_df"].copy()

    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">Export</div>
            <div class="subtitle">Download rankings and sampled comments.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    summary = dv.copy()
    summary["generated_at"] = res["generated_at"]
    summary = summary[
        [
            "rank",
            "video_id",
            "title",
            "channel",
            "published_at",
            "views",
            "likes",
            "comment_count",
            "sampled_comments",
            "engagement_rate",
            "avg_sentiment",
            "pos_pct",
            "neu_pct",
            "neg_pct",
            "relevance_score",
            "freshness_days",
            "final_score",
            "top_keywords",
            "why_this_ranked",
            "video_url",
            "generated_at",
        ]
    ]

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download rankings CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"batsirai_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "Download sampled comments CSV",
            data=dc.to_csv(index=False).encode("utf-8"),
            file_name=f"batsirai_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("")
st.markdown(
    f"""
    <div class="card-soft" style="text-align:center;">
        <div style="font-weight:800;">{APP_NAME} v{APP_VERSION}</div>
        <div class="subtitle">Secure session • {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

logout_c1, logout_c2, logout_c3 = st.columns([1, 1, 1])
with logout_c2:
    if st.button("Logout", use_container_width=True):
        logout()
