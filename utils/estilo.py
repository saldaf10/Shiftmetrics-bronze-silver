"""utils/estilo.py — Paleta y helpers de plotly alineados con el tema."""
from __future__ import annotations
import plotly.graph_objects as go

# Colores de la paleta ShiftMetrics (alineados con theme.py)
BRAND        = "#4f46e5"
BRAND_LT     = "#e0e7ff"
ROJO_ATENCION = "#ef4444"
AMBAR_MEDIO   = "#f59e0b"
VERDE_OK      = "#10b981"
ACCENT        = "#06b6d4"
GRIS_FUERTE   = "#1e293b"
GRIS_MEDIO    = "#64748b"
GRIS_SUAVE    = "#e2e8f0"
AZUL_CORP     = "#4f46e5"
FONDO         = "#ffffff"

PALETA_SEMAFORO = {"alto": ROJO_ATENCION, "medio": AMBAR_MEDIO, "bajo": VERDE_OK}

def aplicar_estilo(fig: go.Figure, alto: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=alto,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=56, b=48),
        font=dict(color=GRIS_FUERTE, family="Inter, -apple-system, sans-serif", size=13),
        title_font=dict(color=GRIS_FUERTE, size=16, family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRIS_SUAVE, borderwidth=1,
                    font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor=GRIS_SUAVE, zeroline=False,
                     title_font=dict(color=GRIS_MEDIO, size=13))
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor=GRIS_SUAVE, zeroline=False,
                     title_font=dict(color=GRIS_MEDIO, size=13))
    return fig

def emoji_riesgo(cat: str) -> str:
    return {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}.get(cat, "⚪")

def color_riesgo(cat: str) -> str:
    return PALETA_SEMAFORO.get(cat, GRIS_MEDIO)
