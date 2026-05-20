"""Кастомный CSS для Streamlit. Подключается через st.markdown."""

CUSTOM_CSS = """
<style>
:root {
    --vc-primary: #4C9AFF;
    --vc-accent: #A78BFA;
    --vc-bg: #0E1117;
    --vc-card: #161A22;
    --vc-card-2: #1C2230;
    --vc-success: #34D399;
    --vc-warning: #F59E0B;
    --vc-danger: #F87171;
    --vc-text-muted: #9CA3AF;
}

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { background: linear-gradient(90deg, var(--vc-primary), var(--vc-accent)); -webkit-background-clip: text; background-clip: text; color: transparent; }

/* Cards / sections */
section.main > div.block-container { padding-top: 2rem; }
[data-testid="stExpander"] {
    background: var(--vc-card);
    border: 1px solid #232a36;
    border-radius: 14px;
    overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--vc-card);
    padding: 0.85rem 1rem;
    border-radius: 14px;
    border: 1px solid #232a36;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--vc-text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Buttons */
.stButton button {
    border-radius: 12px;
    font-weight: 600;
    transition: transform 0.08s ease, box-shadow 0.2s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(76, 154, 255, 0.18);
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--vc-primary), var(--vc-accent));
    border: none;
}
.stDownloadButton button {
    border-radius: 12px;
    font-weight: 500;
}

/* Tabs */
.stTabs [role="tablist"] button {
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--vc-primary), var(--vc-accent));
    color: white !important;
}

/* Recording pulse animation */
@keyframes vc-pulse {
    0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.6); }
    70% { box-shadow: 0 0 0 14px rgba(248, 113, 113, 0); }
    100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
}
.vc-recording-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--vc-danger);
    animation: vc-pulse 1.4s infinite;
    margin-right: 8px;
    vertical-align: middle;
}

/* Status/info boxes */
.stAlert {
    border-radius: 14px;
    border: 1px solid #232a36;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--vc-card);
    border-right: 1px solid #1f2630;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem;
    background: none;
    -webkit-background-clip: initial;
    color: var(--vc-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Text inputs / selects */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    background: var(--vc-card) !important;
}

/* Tooltip on captions */
small, .stCaption { color: var(--vc-text-muted) !important; }

/* Audio player */
audio { border-radius: 10px; width: 100%; }

/* Toggles */
.stCheckbox [data-testid="stCheckbox"] {
    padding: 0.25rem 0;
}
</style>
"""


def inject(st_module) -> None:
    st_module.markdown(CUSTOM_CSS, unsafe_allow_html=True)
