"""utils/theme.py — Tema ShiftMetrics con identidad GitHub/developer."""
import streamlit as st

# Paleta inspirada en GitHub
C = {
    # Core GitHub
    "bg_dark":    "#0d1117",   # github dark bg
    "bg_card":    "#161b22",   # github dark card
    "bg_light":   "#f6f8fa",   # github light bg
    "border_dk":  "#30363d",   # github dark border
    "border_lt":  "#d0d7de",   # github light border
    # Accent
    "green":      "#238636",   # github green (success/merge)
    "green_lt":   "#2ea04333",
    "blue":       "#58a6ff",   # github blue (links)
    "blue_dk":    "#0969da",
    "purple":     "#8957e5",   # github purple
    "purple_lt":  "#8957e522",
    "orange":     "#d29922",   # github warning
    "orange_lt":  "#d2992222",
    "red":        "#f85149",   # github danger
    "red_lt":     "#f8514922",
    # Text
    "text":       "#24292f",   # dark mode off
    "text2":      "#57606a",
    "text3":      "#8b949e",
    "text_inv":   "#c9d1d9",   # on dark bg
    # Brand ShiftMetrics
    "brand":      "#8957e5",   # purple = identidad ShiftMetrics
    "brand_lt":   "#f0ebff",
}

GITHUB_LOGO = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
OCTOCAT_URL = "https://octodex.github.com/images/original.png"

def aplicar_tema(titulo="ShiftMetrics", icono="⚡"):
    st.set_page_config(page_title=titulo, page_icon=icono, layout="wide")
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

    /* Sidebar estilo GitHub dark */
    [data-testid="stSidebar"] {{
        background: {C['bg_dark']};
        border-right: 1px solid {C['border_dk']};
    }}
    [data-testid="stSidebar"] * {{ color: {C['text_inv']} !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color: #ffffff !important; }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {C['text3']} !important; font-size: 14px;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {C['border_dk']} !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {C['bg_light']};
        border: 1px solid {C['border_lt']};
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {C['brand']};
        box-shadow: 0 0 0 1px {C['brand']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {C['text2']} !important;
        font-size: 12px !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{
        color: {C['text']} !important;
        font-weight: 800 !important; font-size: 26px !important;
    }}

    /* Tabs estilo GitHub */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0; border-bottom: 1px solid {C['border_lt']};
        background: transparent; padding: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0;
        font-weight: 600; font-size: 14px;
        padding: 10px 20px; border: 1px solid transparent;
        border-bottom: none; margin-bottom: -1px;
    }}
    .stTabs [aria-selected="true"] {{
        background: white !important;
        border-color: {C['border_lt']} !important;
        border-bottom: 1px solid white !important;
        color: {C['text']} !important;
    }}

    /* Botones */
    .stDownloadButton button {{
        background: {C['green']} !important;
        color: white !important; border: none !important;
        border-radius: 6px; font-weight: 600; padding: 8px 20px;
    }}
    .stDownloadButton button:hover {{ background: #2ea043 !important; }}

    /* DataFrames */
    .stDataFrame th {{
        background: {C['bg_light']} !important;
        color: {C['text']} !important; font-weight: 700 !important;
        font-size: 12px !important; text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 2px solid {C['border_lt']} !important;
    }}

    /* Spacing */
    .block-container {{ padding: 2rem 3rem 3rem; max-width: 1400px; }}
    hr {{ border: none; height: 1px; background: {C['border_lt']}; margin: 1.5rem 0; }}

    /* Selectbox / radio / slider accent */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {C['brand']} !important;
    }}
    .stRadio label {{ font-weight: 500; }}
    </style>
    """, unsafe_allow_html=True)


def hero(titulo, subtitulo, badges_html=""):
    st.markdown(
        f"<div style='background:{C['bg_dark']}; border:1px solid {C['border_dk']}; "
        f"border-radius:12px; padding:36px 40px; margin-bottom:20px;'>"
        f"<div style='display:flex; align-items:center; gap:16px;'>"
        f"<img src='{GITHUB_LOGO}' width='48' style='border-radius:50%;'>"
        f"<div>"
        f"<h1 style='color:white; margin:0; font-size:36px; line-height:1;'>{titulo}</h1>"
        f"<p style='color:{C['text3']}; font-size:16px; margin:6px 0 0;'>{subtitulo}</p>"
        f"</div></div>"
        f"{'<div style=\"margin-top:14px;\">' + badges_html + '</div>' if badges_html else ''}"
        f"</div>",
        unsafe_allow_html=True
    )


def badge(texto, tipo="brand"):
    colores = {
        "brand":   (C["purple_lt"], C["purple"]),
        "success": (C["green_lt"],  C["green"]),
        "warning": (C["orange_lt"], C["orange"]),
        "danger":  (C["red_lt"],    C["red"]),
        "info":    ("#ddf4ff",      C["blue_dk"]),
    }
    bg, fg = colores.get(tipo, colores["brand"])
    return (f"<span style='background:{bg}; color:{fg}; padding:3px 10px; "
            f"border-radius:20px; font-size:12px; font-weight:600; "
            f"border:1px solid {fg}33;'>{texto}</span>")


def nota(texto, icono="💡", tipo="brand"):
    borde = C.get(tipo, C["brand"])
    return st.markdown(
        f"<div style='background:{C['bg_light']}; border-left:3px solid {borde}; "
        f"padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; "
        f"font-size:14px; color:{C['text']}; line-height:1.6;'>"
        f"{icono} {texto}</div>",
        unsafe_allow_html=True
    )


def separador():
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


def semaforo_card(emoji, titulo, valor, detalle, color):
    return (
        f"<div style='background:white; border:1px solid {C['border_lt']}; "
        f"border-radius:12px; padding:20px; text-align:center;'>"
        f"<div style='font-size:36px; margin-bottom:6px;'>{emoji}</div>"
        f"<div style='font-weight:700; color:{color}; font-size:15px;'>{titulo}</div>"
        f"<div style='font-size:24px; font-weight:800; color:{C['text']}; margin:4px 0;'>{valor}</div>"
        f"<div style='color:{C['text3']}; font-size:13px;'>{detalle}</div>"
        f"</div>"
    )
