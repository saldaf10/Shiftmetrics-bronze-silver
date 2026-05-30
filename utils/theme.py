"""utils/theme.py — CSS global + helpers visuales con identidad ShiftMetrics.
Importar en cada pagina con: from utils.theme import aplicar_tema
"""
import streamlit as st

# Paleta con personalidad — no gris corporativo sino tech moderno
COLORS = {
    "brand":     "#4f46e5",   # indigo vibrante — identidad ShiftMetrics
    "brand_lt":  "#e0e7ff",   # indigo claro para fondos
    "danger":    "#ef4444",   # rojo vivo
    "danger_lt": "#fef2f2",
    "warning":   "#f59e0b",   # ambar
    "warning_lt":"#fffbeb",
    "success":   "#10b981",   # verde
    "success_lt":"#ecfdf5",
    "text":      "#1e293b",   # slate oscuro
    "text2":     "#64748b",   # slate medio
    "text3":     "#94a3b8",   # slate claro
    "bg":        "#ffffff",
    "bg2":       "#f8fafc",   # fondo sutil
    "bg3":       "#f1f5f9",   # fondo cards
    "border":    "#e2e8f0",
    "accent":    "#06b6d4",   # cyan para highlights
}

def aplicar_tema(titulo: str = "ShiftMetrics", icono: str = "⚡"):
    """Inyecta CSS global. Llamar al inicio de cada pagina."""
    st.set_page_config(page_title=titulo, page_icon=icono, layout="wide")
    st.markdown(f"""
    <style>
    /* ─── Reset y tipografia ─────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

    /* ─── Sidebar ────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['brand']} 0%, #312e81 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: #e0e7ff !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {{
        color: #c7d2fe !important;
        font-size: 14px;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #ffffff !important;
    }}

    /* ─── Metric cards ───────────────────────────── */
    [data-testid="stMetric"] {{
        background: {COLORS['bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 12px rgba(79,70,229,0.12);
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['text2']} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
        font-weight: 800 !important;
        font-size: 28px !important;
    }}

    /* ─── DataFrames / tablas ─────────────────────── */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}
    .stDataFrame th {{
        background: {COLORS['brand']} !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 12px 16px !important;
    }}
    .stDataFrame td {{
        padding: 10px 16px !important;
        font-size: 14px !important;
        border-bottom: 1px solid {COLORS['border']} !important;
    }}

    /* ─── Tabs ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {COLORS['bg2']};
        border-radius: 12px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
        padding: 10px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['brand']} !important;
        color: white !important;
    }}

    /* ─── Sliders ────────────────────────────────── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {COLORS['brand']} !important;
    }}

    /* ─── Botones ────────────────────────────────── */
    .stDownloadButton button {{
        background: {COLORS['brand']} !important;
        color: white !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 24px;
    }}
    .stDownloadButton button:hover {{
        background: #4338ca !important;
    }}

    /* ─── Info/warning boxes ─────────────────────── */
    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    /* ─── Spacing general ────────────────────────── */
    .block-container {{
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
    }}

    /* ─── Dividers ───────────────────────────────── */
    hr {{
        border: none;
        height: 1px;
        background: {COLORS['border']};
        margin: 1.5rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)


def card(contenido: str, color: str = "bg3", borde: str = "border"):
    """Tarjeta HTML con estilo."""
    st.markdown(
        f"<div style='background:{COLORS[color]}; border:1px solid {COLORS[borde]}; "
        f"border-radius:14px; padding:20px 24px; margin:8px 0;'>"
        f"{contenido}</div>",
        unsafe_allow_html=True
    )

def badge(texto: str, tipo: str = "brand"):
    """Badge inline."""
    bg = COLORS.get(f"{tipo}_lt", COLORS["brand_lt"])
    fg = COLORS.get(tipo, COLORS["brand"])
    return (f"<span style='background:{bg}; color:{fg}; padding:4px 12px; "
            f"border-radius:20px; font-size:12px; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:0.04em;'>{texto}</span>")

def stat_grande(valor: str, label: str, color: str = "brand"):
    """Numero grande con label debajo."""
    c = COLORS.get(color, COLORS["brand"])
    st.markdown(
        f"<div style='text-align:center; padding:16px 0;'>"
        f"<div style='font-size:48px; font-weight:800; color:{c}; line-height:1;'>{valor}</div>"
        f"<div style='font-size:14px; color:{COLORS['text2']}; margin-top:6px; font-weight:600;'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

def nota_lateral(texto: str, icono: str = "💡"):
    """Nota con fondo sutil y borde izquierdo."""
    st.markdown(
        f"<div style='background:{COLORS['bg3']}; border-left:4px solid {COLORS['brand']}; "
        f"padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; "
        f"font-size:14px; color:{COLORS['text']}; line-height:1.6;'>"
        f"{icono} {texto}</div>",
        unsafe_allow_html=True
    )

def separador():
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
