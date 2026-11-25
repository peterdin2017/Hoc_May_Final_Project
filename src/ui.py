"""
ui.py - Giao diện dùng chung: CSS, banner đầu trang, huy hiệu mức rủi ro.
Chỉ chứa phần trình bày, không có logic xử lý dữ liệu.
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stSidebar"] a {
    font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif;
}
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
/* Ẩn nút Deploy */
[data-testid="stAppDeployButton"] { display: none; }

/* Ô số liệu (st.metric) thành thẻ */
[data-testid="stMetric"] {
    background: color-mix(in srgb, currentColor 5%, transparent);
    border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 14px; padding: 16px 18px;
}
[data-testid="stMetricLabel"] p { opacity: .75; font-size: .85rem; }
[data-testid="stMetricValue"] { font-weight: 700; }

/* Tab */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 8px 16px; }

/* Banner trang chủ */
.hero {
    background-color: #0B1F33; border-radius: 18px; padding: 36px 40px; margin-bottom: 1.4rem;
    background-image:
        linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
    background-size: 28px 28px;
}
.hero-title { color: #FFFFFF; font-size: 2.1rem; font-weight: 700; line-height: 1.25; margin-bottom: .6rem; }
.hero-text { color: #B8C7D6; font-size: 1.02rem; line-height: 1.65; max-width: 62ch; margin-bottom: 1.1rem; }
.chip {
    display: inline-block; padding: 4px 12px; margin: 0 8px 6px 0; border-radius: 999px;
    background: rgba(94, 234, 212, .12); border: 1px solid rgba(94, 234, 212, .35);
    color: #5EEAD4; font-size: .8rem; font-weight: 500;
}

/* Tiêu đề các trang bên trong */
.ph { margin-bottom: 1.2rem; padding-bottom: .9rem; border-bottom: 1px solid color-mix(in srgb, currentColor 16%, transparent); }
.ph-title { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em; }
.ph-sub { opacity: .7; font-size: .98rem; margin-top: 2px; }

/* Thẻ điều hướng ở trang chủ */
.card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
.card-text { opacity: .75; font-size: .92rem; line-height: 1.55; min-height: 4.2em; }
.st-key-card_predict { border-color: #0F766E !important; box-shadow: 0 0 0 1px #0F766E; }

/* Huy hiệu mức rủi ro */
.badge { display: inline-block; padding: 7px 18px; border-radius: 999px; font-weight: 700; font-size: 1rem; }
.b-low  { background: #DCFCE7; color: #166534; }
.b-mid  { background: #FEF3C7; color: #92400E; }
.b-high { background: #FEE2E2; color: #991B1B; }
.legend-row { opacity: .85; font-size: .9rem; line-height: 1.9; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
</style>
"""


def inject_css():
    """Gọi 1 lần ở app.py - áp dụng cho mọi trang."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, text, chips=()):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(
        f'<div class="hero"><div class="hero-title">{title}</div>'
        f'<div class="hero-text">{text}</div>{chips_html}</div>',
        unsafe_allow_html=True,
    )


def page_header(title, subtitle=""):
    sub = f'<div class="ph-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="ph"><div class="ph-title">{title}</div>{sub}</div>', unsafe_allow_html=True)


def risk_badge(level, box_type):
    """box_type: success / warning / error (giống classify_risk)."""
    css = {"success": "b-low", "warning": "b-mid"}.get(box_type, "b-high")
    st.markdown(f'<span class="badge {css}">Mức rủi ro: {level}</span>', unsafe_allow_html=True)