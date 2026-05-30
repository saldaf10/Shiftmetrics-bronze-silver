"""utils/theme.py — Tema ShiftMetrics dark mode radical.
Inspirado en dashboards F1/monitoring: fondo oscuro total, acentos neon,
numeros grandes, glassmorphism, tipografia bold.
"""
import streamlit as st

# Paleta dark + neon accents
C = {
    # Fondos
    "bg":         "#0a0a0f",     # negro profundo
    "bg2":        "#111118",     # card bg
    "bg3":        "#1a1a24",     # card hover / elevated
    "bg_glass":   "rgba(20,20,30,0.7)",
    # Bordes
    "border":     "#2a2a3a",
    "border_glow":"#8957e530",
    # Brand
    "brand":      "#8957e5",     # purple neon
    "brand_glow": "#8957e540",
    "brand_lt":   "#c4b5fd",
    # Accent
    "cyan":       "#06b6d4",
    "cyan_glow":  "#06b6d430",
    # Semaforo
    "green":      "#22c55e",
    "green_bg":   "#22c55e18",
    "amber":      "#f59e0b",
    "amber_bg":   "#f59e0b18",
    "red":        "#ef4444",
    "red_bg":     "#ef444418",
    # Texto
    "text":       "#e2e8f0",
    "text2":      "#94a3b8",
    "text3":      "#64748b",
    "text_bright":"#ffffff",
    # Gradientes
    "grad_brand": "linear-gradient(135deg, #8957e5 0%, #06b6d4 100%)",
    "grad_dark":  "linear-gradient(180deg, #111118 0%, #0a0a0f 100%)",
    "grad_card":  "linear-gradient(135deg, rgba(137,87,229,0.08) 0%, rgba(6,182,212,0.05) 100%)",
}

def aplicar_tema(titulo="ShiftMetrics", icono="⚡"):
    st.set_page_config(page_title=titulo, page_icon=icono, layout="wide")
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* ═══ DARK EVERYTHING ═══ */
    html, body, [class*="css"],
    .stApp, .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stMain"] {{
        background-color: {C['bg']} !important;
        color: {C['text']} !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}

    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

    .block-container {{ padding: 1.5rem 2.5rem 3rem; max-width: 1500px; }}

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        background: {C['bg2']} !important;
        border-right: 1px solid {C['border']} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {C['text2']} !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color: {C['text_bright']} !important; }}
    [data-testid="stSidebar"] hr {{ border-color: {C['border']} !important; }}
    [data-testid="stSidebar"] .stAlert {{ background: {C['bg3']} !important; border: 1px solid {C['border']}; }}

    /* ═══ HEADINGS ═══ */
    h1 {{ color: {C['text_bright']} !important; font-weight: 900 !important; }}
    h2 {{ color: {C['text_bright']} !important; font-weight: 800 !important; }}
    h3 {{ color: {C['text']} !important; font-weight: 700 !important; }}
    p, li {{ color: {C['text2']} !important; }}

    /* ═══ METRIC CARDS ═══ */
    [data-testid="stMetric"] {{
        background: {C['bg2']} !important;
        border: 1px solid {C['border']} !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        transition: all 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {C['brand']} !important;
        box-shadow: 0 0 20px {C['brand_glow']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {C['text3']} !important;
        font-size: 11px !important; font-weight: 700 !important;
        text-transform: uppercase; letter-spacing: 0.1em;
    }}
    [data-testid="stMetricValue"] {{
        color: {C['text_bright']} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 800 !important; font-size: 28px !important;
    }}

    /* ═══ TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: {C['bg2']}; border-radius: 10px; padding: 4px;
        border: 1px solid {C['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px; font-weight: 600; font-size: 13px;
        padding: 8px 18px; color: {C['text3']} !important; background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C['brand']} !important;
        color: white !important;
    }}

    /* ═══ DATAFRAMES ═══ */
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
    .stDataFrame, .stDataFrame * {{ background: {C['bg2']} !important; color: {C['text']} !important; }}
    .stDataFrame th {{
        background: {C['bg3']} !important; color: {C['text_bright']} !important;
        font-weight: 700 !important; font-size: 11px !important;
        text-transform: uppercase; letter-spacing: 0.06em;
        border-bottom: 2px solid {C['brand']} !important;
    }}
    .stDataFrame td {{ border-bottom: 1px solid {C['border']} !important; }}

    /* ═══ INPUTS ═══ */
    .stSelectbox, .stMultiSelect, .stNumberInput, .stTextInput {{
        color: {C['text']} !important;
    }}
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background: {C['bg3']} !important;
        border-color: {C['border']} !important;
        color: {C['text']} !important;
    }}

    /* ═══ SLIDER ═══ */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {C['brand']} !important;
    }}
    .stSlider label {{ color: {C['text2']} !important; }}

    /* ═══ RADIO ═══ */
    .stRadio label {{ color: {C['text']} !important; font-weight: 500; }}
    .stRadio [data-testid="stWidgetLabel"] {{ color: {C['text2']} !important; }}

    /* ═══ BUTTONS ═══ */
    .stDownloadButton button {{
        background: {C['brand']} !important;
        color: white !important; border: none !important;
        border-radius: 8px; font-weight: 700;
    }}
    .stDownloadButton button:hover {{ opacity: 0.85; }}

    /* ═══ ALERTS ═══ */
    [data-testid="stAlert"] {{
        background: {C['bg3']} !important;
        border: 1px solid {C['border']} !important;
        border-radius: 10px;
        color: {C['text']} !important;
    }}

    /* ═══ DIVIDERS ═══ */
    hr {{ border: none; height: 1px; background: {C['border']}; margin: 1.2rem 0; }}

    /* ═══ EXPANDER ═══ */
    .streamlit-expanderHeader {{ background: {C['bg2']} !important; color: {C['text']} !important; }}
    .streamlit-expanderContent {{ background: {C['bg2']} !important; }}

    /* ═══ PLOTLY ═══ */
    .js-plotly-plot .plotly {{ background: transparent !important; }}
    </style>
    """, unsafe_allow_html=True)


def hero(titulo, subtitulo, badges_html=""):
    """Hero block con gradiente brand."""
    st.markdown(
        f"<div style='background:{C['grad_card']}; border:1px solid {C['border']}; "
        f"border-radius:16px; padding:32px 36px; margin-bottom:16px;'>"
        f"<div style='font-size:12px; color:{C['brand_lt']}; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:0.15em; margin-bottom:8px;'>SPRINT DEFECT PREDICTION</div>"
        f"<h1 style='margin:0; font-size:40px; background:{C['grad_brand']}; "
        f"-webkit-background-clip:text; -webkit-text-fill-color:transparent; "
        f"background-clip:text;'>{titulo}</h1>"
        f"<p style='color:{C['text2']}; font-size:15px; margin:8px 0 0; max-width:600px;'>{subtitulo}</p>"
        f"{'<div style=\"margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;\">' + badges_html + '</div>' if badges_html else ''}"
        f"</div>",
        unsafe_allow_html=True
    )


def badge(texto, tipo="brand"):
    paleta = {
        "brand":   (C["brand"]+"18", C["brand_lt"]),
        "success": (C["green_bg"],   C["green"]),
        "warning": (C["amber_bg"],   C["amber"]),
        "danger":  (C["red_bg"],     C["red"]),
        "info":    (C["cyan_glow"],  C["cyan"]),
    }
    bg, fg = paleta.get(tipo, paleta["brand"])
    return (f"<span style='background:{bg}; color:{fg}; padding:4px 12px; "
            f"border-radius:6px; font-size:12px; font-weight:700; "
            f"font-family:JetBrains Mono,monospace; "
            f"border:1px solid {fg}30;'>{texto}</span>")


def nota(texto, icono="", tipo="brand"):
    borde = {"brand":C["brand"],"success":C["green"],"warning":C["amber"],"danger":C["red"]}.get(tipo,C["brand"])
    st.markdown(
        f"<div style='background:{C['bg2']}; border-left:3px solid {borde}; "
        f"padding:14px 18px; border-radius:0 10px 10px 0; margin:10px 0; "
        f"font-size:14px; color:{C['text2']}; line-height:1.6; "
        f"border:1px solid {C['border']}; border-left:3px solid {borde};'>"
        f"{icono + ' ' if icono else ''}{texto}</div>",
        unsafe_allow_html=True
    )


def stat_card(valor, label, sublabel="", color="brand"):
    """Numero grande estilo F1 timing."""
    c = {"brand":C["brand"],"green":C["green"],"amber":C["amber"],"red":C["red"],"cyan":C["cyan"]}.get(color,C["brand"])
    st.markdown(
        f"<div style='background:{C['bg2']}; border:1px solid {C['border']}; "
        f"border-radius:14px; padding:24px; text-align:center;'>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:42px; "
        f"font-weight:900; color:{c}; line-height:1;'>{valor}</div>"
        f"<div style='font-size:13px; color:{C['text2']}; margin-top:8px; "
        f"font-weight:600; text-transform:uppercase; letter-spacing:0.06em;'>{label}</div>"
        f"{'<div style=\"font-size:12px; color:'+C['text3']+'; margin-top:4px;\">'+sublabel+'</div>' if sublabel else ''}"
        f"</div>",
        unsafe_allow_html=True
    )


def separador():
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


def seccion_titulo(titulo, subtitulo=""):
    st.markdown(
        f"<div style='margin:8px 0 16px;'>"
        f"<h2 style='margin:0; font-size:22px;'>{titulo}</h2>"
        f"{'<p style=\"color:'+C['text3']+'; font-size:14px; margin:4px 0 0;\">'+subtitulo+'</p>' if subtitulo else ''}"
        f"</div>",
        unsafe_allow_html=True
    )
