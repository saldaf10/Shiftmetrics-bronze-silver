"""utils/estilo.py — Plotly helpers alineados con tema GitHub."""
import plotly.graph_objects as go

BRAND        = "#8957e5"
ROJO_ATENCION = "#f85149"
AMBAR_MEDIO   = "#d29922"
VERDE_OK      = "#238636"
ACCENT        = "#58a6ff"
AZUL_CORP     = "#0969da"
GRIS_FUERTE   = "#24292f"
GRIS_MEDIO    = "#57606a"
GRIS_SUAVE    = "#d0d7de"
FONDO         = "#ffffff"
PALETA_SEMAFORO = {"alto": ROJO_ATENCION, "medio": AMBAR_MEDIO, "bajo": VERDE_OK}

def aplicar_estilo(fig, alto=420):
    fig.update_layout(
        template="plotly_white", height=alto,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=56, b=48),
        font=dict(color=GRIS_FUERTE, family="Inter, -apple-system, sans-serif", size=13),
        title_font=dict(color=GRIS_FUERTE, size=16),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=GRIS_SUAVE, borderwidth=1, font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor="#f0f0f0", linecolor=GRIS_SUAVE, zeroline=False,
                     title_font=dict(color=GRIS_MEDIO, size=13))
    fig.update_yaxes(gridcolor="#f0f0f0", linecolor=GRIS_SUAVE, zeroline=False,
                     title_font=dict(color=GRIS_MEDIO, size=13))
    return fig

def emoji_riesgo(cat): return {"alto":"🔴","medio":"🟡","bajo":"🟢"}.get(cat,"⚪")
def color_riesgo(cat): return PALETA_SEMAFORO.get(cat, GRIS_MEDIO)
