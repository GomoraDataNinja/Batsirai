import streamlit as st
import pandas as pd
import re
import html
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import numpy as np
import time
import os
import hashlib
import warnings
import json
from functools import lru_cache


warnings.filterwarnings("ignore")

APP_VERSION = "3.7.0"
APP_NAME = "Batsirai"
DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "production")
SESSION_TIMEOUT_MINUTES = 60

st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="📊",
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

# Geographic coordinates for regions
REGION_COORDINATES = {
    "ZW": {"lat": -19.0154, "lon": 29.1549, "radius": "250km", "name": "Zimbabwe", "code": "ZW"},
    "ZA": {"lat": -30.5595, "lon": 22.9375, "radius": "500km", "name": "South Africa", "code": "ZA"},
    "GB": {"lat": 55.3781, "lon": -3.4360, "radius": "300km", "name": "United Kingdom", "code": "GB"},
    "US": {"lat": 37.0902, "lon": -95.7129, "radius": "500km", "name": "United States", "code": "US"},
    "IN": {"lat": 20.5937, "lon": 78.9629, "radius": "500km", "name": "India", "code": "IN"},
    "NG": {"lat": 9.0820, "lon": 8.6753, "radius": "400km", "name": "Nigeria", "code": "NG"},
    "KE": {"lat": -1.2864, "lon": 36.8172, "radius": "300km", "name": "Kenya", "code": "KE"},
    "GH": {"lat": 7.9465, "lon": -1.0232, "radius": "300km", "name": "Ghana", "code": "GH"},
    "AU": {"lat": -25.2744, "lon": 133.7751, "radius": "500km", "name": "Australia", "code": "AU"},
    "CA": {"lat": 56.1304, "lon": -106.3468, "radius": "500km", "name": "Canada", "code": "CA"},
    "FR": {"lat": 46.2276, "lon": 2.2137, "radius": "500km", "name": "France", "code": "FR"},
    "DE": {"lat": 51.1657, "lon": 10.4515, "radius": "500km", "name": "Germany", "code": "DE"},
}

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

        html {{
            color-scheme: light !important;
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

        div[data-baseweb="layer"],
        div[data-baseweb="layer"] > div {{
            background: transparent !important;
            color: var(--text) !important;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div {{
            background: #ffffff !important;
            color: var(--text) !important;
            border-radius: 14px !important;
            border: 1px solid var(--border2) !important;
            box-shadow: 0 12px 28px rgba(0,0,0,0.10) !important;
        }}

        div[data-baseweb="menu"],
        div[data-baseweb="menu"] > div,
        div[data-baseweb="menu"] > div > div {{
            background: #ffffff !important;
            color: var(--text) !important;
        }}

        ul[role="listbox"],
        div[role="listbox"] {{
            background: #ffffff !important;
            color: var(--text) !important;
        }}

        li[role="option"],
        div[role="option"] {{
            background: #ffffff !important;
            color: var(--text) !important;
        }}

        li[role="option"] * ,
        div[role="option"] * {{
            color: var(--text) !important;
        }}

        li[role="option"]:hover,
        div[role="option"]:hover {{
            background: rgba(215,30,40,0.08) !important;
        }}

        li[role="option"][aria-selected="true"],
        div[role="option"][aria-selected="true"] {{
            background: rgba(215,30,40,0.12) !important;
        }}

        div[data-baseweb="tooltip"],
        div[data-baseweb="tooltip"] > div {{
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border2) !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 22px rgba(0,0,0,0.10) !important;
        }}

        div[data-baseweb="calendar"],
        div[data-baseweb="calendar"] * {{
            background: #ffffff !important;
            color: var(--text) !important;
        }}

        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] div,
        div[data-baseweb="popover"] p,
        div[data-baseweb="menu"] span,
        div[data-baseweb="menu"] div,
        div[data-baseweb="menu"] p {{
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

        a, a:visited {{
            color: var(--accent) !important;
            text-decoration: none !important;
            font-weight: 750 !important;
        }}
        a:hover {{
            color: var(--accent2) !important;
            text-decoration: underline !important;
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
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "api_error" not in st.session_state:
    st.session_state.api_error = None

def login_screen():
    st.markdown('<div style="height: 1.8rem;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.25, 1])
    with c2:
        st.markdown(
            f"""
            <div class="card" style="margin-top: 10vh;">
                <div class="title" style="text-align:center;">{APP_NAME}</div>
                <div class="subtitle" style="text-align:center;">Sign in to continue.</div>
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

@st.cache_resource
def youtube_client():
    """Initialize YouTube client with caching to avoid rebuilding"""
    try:
        api_key = st.secrets["youtube_api_key"]
    except Exception:
        api_key = None

    if not api_key or not str(api_key).strip():
        st.session_state.api_error = "Missing YouTube API key. Please add youtube_api_key to secrets."
        return None

    try:
        return build("youtube", "v3", developerKey=str(api_key).strip())
    except Exception as e:
        st.session_state.api_error = f"Failed to initialize YouTube client: {str(e)}"
        return None

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
    attempts = 0
    max_attempts = 2

    while len(all_comments) < max_comments and attempts < max_attempts:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(all_comments)),
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()
            st.session_state.api_call_count += 1

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

        except HttpError as e:
            if e.resp.status == 403:
                break
            elif "commentsDisabled" in str(e):
                break
            attempts += 1
            time.sleep(1)
        except Exception:
            attempts += 1
            time.sleep(1)

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

def search_videos_by_topic(youtube, topic, max_videos=30, order="relevance", published_after=None, region_code=None, location_filter=None):
    """
    Enhanced search that mimics YouTube's search behavior
    """
    results = []
    next_page_token = None
    attempts = 0
    max_attempts = 2
    
    # Clean and enhance the search query
    search_query = topic.strip()
    
    # Add location context if specified
    if location_filter and location_filter in REGION_COORDINATES:
        location_name = REGION_COORDINATES[location_filter]["name"]
        # Add location to query for better relevance
        search_query = f"{topic} {location_name}"
    
    # Build search parameters - similar to YouTube's search
    search_params = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "maxResults": min(50, max_videos),
        "order": order,
        "safeSearch": "none",
        "videoDuration": "any",  # Include all durations
        "videoType": "any",      # Include all video types
    }
    
    if published_after:
        search_params["publishedAfter"] = published_after
    
    # Location-based filtering
    if location_filter and location_filter in REGION_COORDINATES:
        coords = REGION_COORDINATES[location_filter]
        search_params["location"] = f"{coords['lat']},{coords['lon']}"
        search_params["locationRadius"] = coords['radius']
        search_params["regionCode"] = location_filter
    elif region_code:
        search_params["regionCode"] = region_code

    while len(results) < max_videos and attempts < max_attempts:
        try:
            if next_page_token:
                search_params["pageToken"] = next_page_token
            
            request = youtube.search().list(**search_params)
            resp = request.execute()
            st.session_state.api_call_count += 1
            
            # Track what search returned
            st.session_state.last_search_results = len(resp.get("items", []))
            
            for item in resp.get("items", []):
                vid = item.get("id", {}).get("videoId")
                sn = item.get("snippet", {})
                if not vid:
                    continue
                
                results.append({
                    "video_id": vid,
                    "title": sn.get("title", ""),
                    "description": sn.get("description", ""),
                    "channel": sn.get("channelTitle", ""),
                    "channel_id": sn.get("channelId", ""),
                    "published_at": sn.get("publishedAt", ""),
                    "thumbnails": sn.get("thumbnails", {}),
                })
                
                if len(results) >= max_videos:
                    break

            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        except HttpError as e:
            if e.resp.status == 403:
                st.session_state.api_error = "YouTube API quota exceeded. Please try again later."
                break
            attempts += 1
            time.sleep(1)
        except Exception as e:
            attempts += 1
            time.sleep(1)

    return results

def fetch_video_stats(youtube, video_ids, max_retries=2):
    out = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        retries = 0
        while retries < max_retries:
            try:
                resp = youtube.videos().list(
                    part="snippet,statistics,contentDetails", 
                    id=",".join(chunk)
                ).execute()
                st.session_state.api_call_count += 1
                
                for item in resp.get("items", []):
                    vid = item.get("id")
                    stats = item.get("statistics", {}) or {}
                    sn = item.get("snippet", {}) or {}
                    cd = item.get("contentDetails", {}) or {}
                    
                    # Get channel country
                    channel_id = sn.get("channelId", "")
                    channel_country = None
                    if channel_id:
                        try:
                            channel_resp = youtube.channels().list(
                                part="snippet",
                                id=channel_id
                            ).execute()
                            st.session_state.api_call_count += 1
                            if channel_resp.get("items"):
                                channel_country = channel_resp["items"][0].get("snippet", {}).get("country", "")
                        except:
                            pass
                    
                    out[vid] = {
                        "views": int(stats.get("viewCount", 0) or 0),
                        "likes": int(stats.get("likeCount", 0) or 0),
                        "comment_count": int(stats.get("commentCount", 0) or 0),
                        "title": sn.get("title", ""),
                        "channel": sn.get("channelTitle", ""),
                        "channel_id": channel_id,
                        "channel_country": channel_country,
                        "published_at": sn.get("publishedAt", ""),
                        "description": sn.get("description", ""),
                        "duration": cd.get("duration", ""),
                    }
                break
            except HttpError as e:
                if e.resp.status == 403:
                    break
                retries += 1
                time.sleep(1)
            except Exception:
                retries += 1
                time.sleep(1)
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
        
    if row.get("channel_country"):
        reasons.append(f"Channel is based in {row.get('channel_country')}.")

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
    top_location = str(top_pick.get("channel_country", "")).strip()

    bullets = []

    if dc is None or dc.empty:
        bullets.append(f"I scanned {scanned} videos for this topic.")
        bullets.append("I could not sample comments for this run, so ranking used engagement, topic match, and recency.")
        bullets.append(f"Best pick right now: {top_title} by {top_channel}.")
        if top_location:
            bullets.append(f"This channel is based in {top_location}.")
        bullets.append("Try increasing Comments per video, or pick a different time window or region.")
        overall = {"sentiment_counts": pd.Series({"Positive": 0, "Neutral": 0, "Negative": 0})}
        return {
            "headline": f"What this topic looks like on YouTube: {topic}",
            "bullets": bullets,
            "overall": overall,
            "top_pick": {"title": top_title, "channel": top_channel, "url": top_url, "location": top_location},
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
    if top_location:
        bullets.append(f"This channel is based in {top_location}.")
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
        "top_pick": {"title": top_title, "channel": top_channel, "url": top_url, "location": top_location},
        "themes": {"common": common_points, "positive": pos_points, "negative": neg_points, "questions": q_points},
    }

def run_topic_analysis(topic, max_videos, comments_per_video, order, time_window_days, region_code, location_filter):
    st.session_state.api_error = None
    
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

    # Clean and enhance the search query
    search_topic = topic
    
    # Add location context if filtering by location
    if location_filter and location_filter in REGION_COORDINATES:
        location_name = REGION_COORDINATES[location_filter]["name"]
        search_topic = f"{topic} {location_name}"

    # Search YouTube like YouTube does
    videos = search_videos_by_topic(
        yt,
        topic=search_topic,
        max_videos=max_videos,
        order=order,
        published_after=published_after,
        region_code=region_code if not location_filter else location_filter,
        location_filter=location_filter,
    )
    
    if not videos:
        if st.session_state.api_error:
            st.error(st.session_state.api_error)
        else:
            st.error(f"No videos found for '{topic}'. Try a different topic or adjust your filters.")
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
        channel_country = meta.get("channel_country") or None

        status.markdown(f"<div class='muted'>Scanning video {idx} of {len(ids)}...</div>", unsafe_allow_html=True)

        comments = []
        if comments_per_video > 0 and not st.session_state.api_error:
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
                "channel_country": channel_country,
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
        "location_filter": location_filter,
        "api_calls": st.session_state.api_call_count,
        "search_query": search_topic,
    }

    res["final_answer"] = build_final_answer(res)
    return res

# Main UI
st.markdown(
    f"""
    <div class="hero" style="text-align:center;">
        <div class="title">{APP_NAME}</div>
        <div class="subtitle">Search any topic and get YouTube results with sentiment analysis - just like YouTube but smarter.</div>
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

if st.session_state.api_error:
    st.warning(f"⚠️ {st.session_state.api_error}")

st.markdown("")

tabs = st.tabs(["🔍 Search", "📊 Top 10", "🔬 Explore", "📥 Export"])

with tabs[0]:
    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">🔍 Search YouTube</div>
            <div class="subtitle">Type any question or topic - just like you would on YouTube.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Search input - similar to YouTube
    c1, c2 = st.columns([3, 1])
    with c1:
        topic = st.text_input(
            "What do you want to find?",
            value=st.session_state.topic_query,
            placeholder="Example: best excel tutor, how to budget money, study tips for exams",
            help="Type your question or topic like you would on YouTube"
        )
    with c2:
        order = st.selectbox(
            "Sort by", 
            options=["relevance", "date", "viewCount", "rating"],
            index=0,
            format_func=lambda x: {
                "relevance": "📊 Relevance",
                "date": "📅 Latest",
                "viewCount": "👁️ Most viewed",
                "rating": "⭐ Top rated"
            }.get(x, x)
        )

    # Advanced filters
    with st.expander("⚙️ Advanced filters", expanded=False):
        r1, r2, r3, r4 = st.columns([1, 1, 1, 1.2])
        with r1:
            max_videos = st.slider("Videos to scan", 5, 50, 20, step=5, 
                                  help="Lower this if you're hitting quota limits")
        with r2:
            comments_per_video = st.slider("Comments per video", 20, 150, 80, step=10,
                                          help="Lower this to save quota")
        with r3:
            time_window_days = st.selectbox(
                "Upload date",
                options=[0, 7, 30, 90, 365],
                index=2,
                format_func=lambda x: "Any time" if x == 0 else f"Last {x} days",
            )
        with r4:
            location_filter = st.selectbox(
                "📍 Location",
                options=["Global"] + list(REGION_COORDINATES.keys()),
                index=0,
                format_func=lambda x: "🌍 Anywhere" if x == "Global" else f"📍 {REGION_COORDINATES[x]['name']}",
            )

    # Search button
    search_col1, search_col2, search_col3 = st.columns([2, 1, 2])
    with search_col1:
        run = st.button("🔍 Search YouTube", use_container_width=True, type="primary")
    with search_col3:
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.topic_results = None
            st.session_state.topic_query = ""
            st.session_state.api_error = None
            safe_rerun()

    if run:
        topic_clean = (topic or "").strip()
        if not topic_clean:
            st.error("Please type a topic or question to search.")
        else:
            st.session_state.topic_query = topic_clean
            st.session_state.api_call_count = 0
            st.session_state.api_error = None
            
            location_display = "Anywhere" if location_filter == "Global" else REGION_COORDINATES[location_filter]["name"]
            
            # Show what we're searching for
            st.info(f"🔍 Searching YouTube for: **{topic_clean}** in **{location_display}**")
            
            with st.spinner(f"Searching YouTube and analyzing results..."):
                res = run_topic_analysis(
                    topic=topic_clean,
                    max_videos=int(max_videos),
                    comments_per_video=int(comments_per_video),
                    order=order,
                    time_window_days=int(time_window_days),
                    region_code=None if location_filter == "Global" else location_filter,
                    location_filter=None if location_filter == "Global" else location_filter,
                )
            
            if res is None:
                if not st.session_state.api_error:
                    st.error(f"No results found for '{topic_clean}'. Try a different topic or adjust your search.")
            else:
                st.session_state.topic_results = res
                st.success(f"✅ Found {len(res['videos_df'])} videos! Check the Top 10 tab for results.")
                safe_rerun()

with tabs[1]:
    res = st.session_state.topic_results
    if not res:
        st.markdown(
            """
            <div class="card-soft">
                <div style="font-size:16px; font-weight:800;">No results yet</div>
                <div class="subtitle">Use the Search tab to find videos on YouTube.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    dv = res["videos_df"].copy()
    topic = res["topic"]
    top10 = dv.head(10).copy()

    location_info = ""
    if res.get("location_filter"):
        location_name = REGION_COORDINATES.get(res.get("location_filter"), {}).get("name", res.get("location_filter"))
        location_info = f" 📍 {location_name}"

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:16px; font-weight:800;">📊 Top 10 results for: {html.escape(topic)}{location_info}</div>
            <div class="subtitle">Ranked by engagement, sentiment, topic match, and recency.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    total_scanned = len(dv)
    avg_sent = float(dv["avg_sentiment"].mean()) if total_scanned else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric'><div class='metric-k'>📹 Videos</div><div class='metric-v'>{total_scanned}</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric'><div class='metric-k'>💬 Avg sentiment</div><div class='metric-v'>{avg_sent:.2f}</div></div>", unsafe_allow_html=True)
    with m3:
        location_display = REGION_COORDINATES.get(res.get("location_filter"), {}).get("name", "🌍 Global") if res.get("location_filter") else "🌍 Global"
        st.markdown(f"<div class='metric'><div class='metric-k'>📍 Location</div><div class='metric-v'>{location_display}</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric'><div class='metric-k'>🔑 API Calls</div><div class='metric-v'>{res.get('api_calls', 0)}</div></div>", unsafe_allow_html=True)

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
            <div style="font-size:16px; font-weight:800;">🤖 What I found</div>
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
                <div style="font-size:16px; font-weight:800;">💬 What people are saying</div>
                <div class="subtitle">Key themes from the comment sections.</div>
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
                        <div class="muted" style="margin-top:6px; font-size:12px;">Mentioned in {pct:.1f}% of comments.</div>
                        <div style="margin-top:8px; font-size:12.5px; line-height:1.55;">💬 "{ex}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        render_points("📌 Common topics", themes.get("common", []))
        render_points("👍 Positive feedback", themes.get("positive", []))
        render_points("👎 Negative feedback", themes.get("negative", []))
        render_points("❓ Questions people ask", themes.get("questions", []))

        if top_pick and str(top_pick.get("url", "")).strip():
            u = html.escape(top_pick.get("url", ""))
            location_display = f"📍 {top_pick.get('location', 'Unknown')}" if top_pick.get('location') else ""
            st.markdown(
                f"""
                <div class="card-soft" style="margin-top: 12px;">
                    <div style="font-weight:800;">🏆 Best pick</div>
                    <div class="subtitle" style="margin-top:6px;">{html.escape(top_pick.get('title',''))}</div>
                    <div class="muted" style="font-size:13px; margin-top:4px;">{html.escape(top_pick.get('channel',''))}</div>
                    <div class="muted" style="font-size:12px; margin-top:2px;">{location_display}</div>
                    <div style="margin-top:10px; font-size:12.5px;">
                        <a href="{u}" target="_blank" rel="noopener">▶️ Watch on YouTube</a>
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
                fig = donut_chart(sc, "💬 Comment sentiment", center)
                st.plotly_chart(fig, use_container_width=True, key="top10_overall_donut")
            else:
                st.markdown(
                    """
                    <div class="card-soft">
                        <div style="font-weight:800;">💬 Sentiment chart</div>
                        <div class="subtitle">No comments sampled for this run.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">📋 Top 10 results</div>
            <div class="subtitle">Full list of the best videos for your search.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    show_cols = [
        "rank",
        "title",
        "channel",
        "channel_country",
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
    t["channel_country"] = t["channel_country"].fillna("Unknown")

    st.dataframe(t, use_container_width=True, height=340)

    st.markdown("")
    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; font-weight:800;">💡 Why these videos rank high</div>
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
        location = str(row.get("channel_country", "Unknown")).strip()
        chip = f"<span class='chip'>#{int(row['rank'])}</span>"
        location_chip = f"<span class='chip'>📍 {location}</span>" if location and location != "Unknown" else ""

        st.markdown(
            f"""
            <div class="card-soft" style="margin-top: 10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
                    <div style="font-weight:800; font-size:15px;">{title}</div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        {location_chip}
                        {chip}
                    </div>
                </div>
                <div class="muted" style="margin-top:6px; font-size:13px;">📺 {channel}</div>
                <div style="margin-top:10px; line-height:1.6; font-size:13px; white-space:pre-wrap;">{html.escape(str(reasons))}</div>
                <div style="margin-top:10px; font-size:12.5px;">
                    <a href="{url}" target="_blank" rel="noopener">▶️ Watch on YouTube</a>
                </div>
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
            <div style="font-size:16px; font-weight:800;">🔬 Explore video details</div>
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
            title="📊 Score distribution (first 30)",
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
            fig2 = donut_chart(s_counts, "💬 Comment sentiment", center)
            st.plotly_chart(fig2, use_container_width=True, key="explore_overall_donut")
        else:
            st.markdown(
                """
                <div class="card-soft">
                    <div style="font-weight:800;">💬 No comments</div>
                    <div class="subtitle">Some videos may block comments or restrict access.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    pick = st.selectbox(
        "🎬 Select a video",
        options=dv["video_id"].tolist(),
        format_func=lambda x: dv.loc[dv["video_id"] == x, "title"].values[0][:80],
        key="explore_video_pick",
    )

    row = dv[dv["video_id"] == pick].iloc[0].to_dict()
    v_url = str(row.get("video_url", "")).strip()
    v_url_safe = html.escape(v_url)
    location = str(row.get("channel_country", "Unknown")).strip()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"<div class='metric'><div class='metric-k'>👁️ Views</div><div class='metric-v'>{int(row.get('views',0)):,}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric'><div class='metric-k'>👍 Likes</div><div class='metric-v'>{int(row.get('likes',0)):,}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric'><div class='metric-k'>💬 Comments</div><div class='metric-v'>{int(row.get('comment_count',0)):,}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric'><div class='metric-k'>🏆 Rank</div><div class='metric-v'>#{int(row.get('rank',0))}</div></div>", unsafe_allow_html=True)
    with k5:
        st.markdown(f"<div class='metric'><div class='metric-k'>📍 Location</div><div class='metric-v'>{location}</div></div>", unsafe_allow_html=True)

    st.markdown("")
    if v_url:
        st.markdown(
            f"""
            <div class="card-soft">
                <div style="font-weight:800;">▶️ Watch now</div>
                <div style="margin-top:8px;"><a href="{v_url_safe}" target="_blank" rel="noopener">Open on YouTube</a></div>
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
                    "Filter by sentiment",
                    options=["Positive", "Neutral", "Negative"],
                    default=["Positive", "Neutral", "Negative"],
                    key="explore_sent_filter",
                )
            with f2:
                sort_by = st.selectbox(
                    "Sort comments",
                    options=["Newest", "Oldest", "Most Likes", "Highest Sentiment", "Lowest Sentiment"],
                    key="explore_sort_by",
                )
            with f3:
                n_show = st.slider("Show rows", 10, 80, 20, key="explore_rows")

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
            <div style="font-size:16px; font-weight:800;">📥 Export data</div>
            <div class="subtitle">Download rankings and sampled comments as CSV files.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    summary = dv.copy()
    summary["generated_at"] = res["generated_at"]
    summary["location_filter"] = res.get("location_filter", "Global")
    summary["search_query"] = res.get("search_query", res.get("topic", ""))
    summary = summary[
        [
            "rank",
            "video_id",
            "title",
            "channel",
            "channel_country",
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
            "location_filter",
            "search_query",
        ]
    ]

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "📥 Download rankings CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"batsirai_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "📥 Download comments CSV",
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
        <div class="subtitle">🔍 YouTube search with sentiment analysis • {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

logout_c1, logout_c2, logout_c3 = st.columns([1, 1, 1])
with logout_c2:
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        
