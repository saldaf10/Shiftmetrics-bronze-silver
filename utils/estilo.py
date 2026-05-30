"""utils/estilo.py — Plotly dark mode."""
import plotly.graph_objects as go

BRAND        = "#8957e5"
ROJO_ATENCION = "#ef4444"
AMBAR_MEDIO   = "#f59e0b"
VERDE_OK      = "#22c55e"
ACCENT        = "#06b6d4"
AZUL_CORP     = "#8957e5"
GRIS_FUERTE   = "#e2e8f0"
GRIS_MEDIO    = "#94a3b8"
GRIS_SUAVE    = "#2a2a3a"
FONDO         = "#0a0a0f"
PALETA_SEMAFORO = {"alto": ROJO_ATENCION, "medio": AMBAR_MEDIO, "bajo": VERDE_OK}

def aplicar_estilo(fig, alto=420):
    fig.update_layout(
        template="plotly_dark", height=alto,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,17,24,0.5)",
        margin=dict(l=48, r=24, t=56, b=48),
        font=dict(color="#e2e8f0", family="Inter, sans-serif", size=13),
        title_font=dict(color="#ffffff", size=16),
        legend=dict(bgcolor="rgba(17,17,24,0.8)", bordercolor="#2a2a3a", borderwidth=1,
                    font=dict(size=12, color="#94a3b8")),
    )
    fig.update_xaxes(gridcolor="#1e1e2e", linecolor="#2a2a3a", zeroline=False,
                     title_font=dict(color="#94a3b8"))
    fig.update_yaxes(gridcolor="#1e1e2e", linecolor="#2a2a3a", zeroline=False,
                     title_font=dict(color="#94a3b8"))
    return fig

def emoji_riesgo(cat): return {"alto":"🔴","medio":"🟡","bajo":"🟢"}.get(cat,"⚪")
def color_riesgo(cat): return PALETA_SEMAFORO.get(cat, GRIS_MEDIO)
