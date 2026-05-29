"""utils/estilo.py — Paleta corporativa y helpers de plotly."""
from __future__ import annotations

import plotly.graph_objects as go

# Jerarquia clara:
#  - rojo intenso = "atencion ya" (riesgo alto, drift critico)
#  - ambar       = "monitorear"   (riesgo medio)
#  - verde       = "todo OK"      (riesgo bajo, calibracion OK)
#  - azul        = identidad corporativa (NO usar para destacar)
#  - grises      = contexto, ejes, fondos
ROJO_ATENCION   = "#dc2626"
AMBAR_MEDIO     = "#f59e0b"
VERDE_OK        = "#10b981"
AZUL_CORP       = "#1e3a8a"
GRIS_FUERTE     = "#374151"
GRIS_MEDIO      = "#9ca3af"
GRIS_SUAVE      = "#e5e7eb"
FONDO           = "#ffffff"

PALETA_SEMAFORO = {
    "alto":  ROJO_ATENCION,
    "medio": AMBAR_MEDIO,
    "bajo":  VERDE_OK,
}


def aplicar_estilo(fig: go.Figure, alto: int = 380) -> go.Figure:
    """Estilo plotly uniforme para todo el dashboard."""
    fig.update_layout(
        template="plotly_white",
        height=alto,
        paper_bgcolor=FONDO,
        plot_bgcolor=FONDO,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(color=GRIS_FUERTE, family="Inter, Segoe UI, sans-serif", size=12),
        title_font=dict(color=GRIS_FUERTE, size=15),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=GRIS_SUAVE, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=GRIS_SUAVE, linecolor=GRIS_MEDIO, zeroline=False,
                     title_font=dict(color=GRIS_FUERTE))
    fig.update_yaxes(gridcolor=GRIS_SUAVE, linecolor=GRIS_MEDIO, zeroline=False,
                     title_font=dict(color=GRIS_FUERTE))
    return fig


def emoji_riesgo(categoria: str) -> str:
    return {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}.get(categoria, "⚪")


def color_riesgo(categoria: str) -> str:
    return PALETA_SEMAFORO.get(categoria, GRIS_MEDIO)
