"""Pagina: Salud del modelo
Pregunta unica: ¿El modelo se esta deteriorando? ¿Hay que reentrenar?
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.data import (cargar_drift_yearly, cargar_drift_psi,
                         cargar_metricas, feature_descripcion)
from utils.estilo import (aplicar_estilo, ROJO_ATENCION, AMBAR_MEDIO, VERDE_OK,
                          AZUL_CORP, GRIS_FUERTE, GRIS_MEDIO, GRIS_SUAVE)


st.set_page_config(page_title="Salud del modelo · ShiftMetrics",
                   page_icon="📡", layout="wide")


st.markdown("## 📡 Salud del modelo")
st.markdown(
    "<p style='color:#6b7280; font-size:16px;'>"
    "Si el desempeño se esta cayendo o si hay variables "
    "que ya no se parecen a los datos con los que "
    "se entreno, puede que toque reentrenar."
    "</p>",
    unsafe_allow_html=True
)
st.divider()


drift_yearly = cargar_drift_yearly()
drift_psi    = cargar_drift_psi()
metricas     = cargar_metricas()
desc         = feature_descripcion()


# ========================================================
# Indicadores rapidos
# ========================================================
st.markdown("### Diagnóstico rápido")

# Definicion de semaforos
f2_test = metricas["f2"]
psi_max = drift_psi["psi"].max() if len(drift_psi) > 0 else 0
n_drift_alto = (drift_psi["psi"] > 0.25).sum() if len(drift_psi) > 0 else 0

# F2 reciente vs F2 de entrenamiento (referencia ~0.97)
f2_referencia = 0.97
delta_f2 = f2_test - f2_referencia

if delta_f2 > -0.02:
    estado_f2, emoji_f2, color_f2 = "Estable", "🟢", VERDE_OK
elif delta_f2 > -0.05:
    estado_f2, emoji_f2, color_f2 = "Bajo monitoreo", "🟡", AMBAR_MEDIO
else:
    estado_f2, emoji_f2, color_f2 = "Atención", "🔴", ROJO_ATENCION

if psi_max < 0.10:
    estado_drift, emoji_drift, color_drift = "Estable", "🟢", VERDE_OK
elif psi_max < 0.25:
    estado_drift, emoji_drift, color_drift = "Bajo monitoreo", "🟡", AMBAR_MEDIO
else:
    estado_drift, emoji_drift, color_drift = "Atención", "🔴", ROJO_ATENCION

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"<div style='background:#ffffff; border:1px solid #e5e7eb; "
        f"border-radius:12px; padding:18px; text-align:center;'>"
        f"<div style='color:#6b7280; font-size:13px;'>Desempeño reciente</div>"
        f"<div style='font-size:38px; font-weight:800; color:{color_f2}; line-height:1.1;'>"
        f"{emoji_f2}</div>"
        f"<div style='font-weight:600; color:{color_f2}; font-size:16px; margin-top:6px;'>"
        f"{estado_f2}</div>"
        f"<div style='color:#6b7280; font-size:13px; margin-top:6px;'>"
        f"F2 = {f2_test:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div style='background:#ffffff; border:1px solid #e5e7eb; "
        f"border-radius:12px; padding:18px; text-align:center;'>"
        f"<div style='color:#6b7280; font-size:13px;'>Drift en variables</div>"
        f"<div style='font-size:38px; font-weight:800; color:{color_drift}; line-height:1.1;'>"
        f"{emoji_drift}</div>"
        f"<div style='font-weight:600; color:{color_drift}; font-size:16px; margin-top:6px;'>"
        f"{estado_drift}</div>"
        f"<div style='color:#6b7280; font-size:13px; margin-top:6px;'>"
        f"PSI máximo = {psi_max:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

with col3:
    if n_drift_alto == 0:
        emoji_acc, color_acc, txt_acc = "🟢", VERDE_OK, "Sin acción"
    elif n_drift_alto <= 2:
        emoji_acc, color_acc, txt_acc = "🟡", AMBAR_MEDIO, "Vigilar"
    else:
        emoji_acc, color_acc, txt_acc = "🔴", ROJO_ATENCION, "Reentrenar"
    st.markdown(
        f"<div style='background:#ffffff; border:1px solid #e5e7eb; "
        f"border-radius:12px; padding:18px; text-align:center;'>"
        f"<div style='color:#6b7280; font-size:13px;'>Variables con drift alto</div>"
        f"<div style='font-size:38px; font-weight:800; color:{color_acc}; line-height:1.1;'>"
        f"{emoji_acc}</div>"
        f"<div style='font-weight:600; color:{color_acc}; font-size:16px; margin-top:6px;'>"
        f"{txt_acc}</div>"
        f"<div style='color:#6b7280; font-size:13px; margin-top:6px;'>"
        f"{n_drift_alto} variables con PSI > 0.25</div>"
        f"</div>",
        unsafe_allow_html=True
    )


st.markdown(" ")
st.divider()


# ========================================================
# Desempeño año a año
# ========================================================
st.markdown("### Evolución del desempeño")

fig_temp = go.Figure()

# Banda de referencia
f2_min_objetivo = 0.90
fig_temp.add_hrect(y0=0.95, y1=1.0, fillcolor=VERDE_OK, opacity=0.08, line_width=0)
fig_temp.add_hrect(y0=f2_min_objetivo, y1=0.95, fillcolor=AMBAR_MEDIO, opacity=0.08, line_width=0)
fig_temp.add_hrect(y0=0.85, y1=f2_min_objetivo, fillcolor=ROJO_ATENCION, opacity=0.08, line_width=0)

# Linea F2 por año
fig_temp.add_trace(go.Scatter(
    x=drift_yearly["año"], y=drift_yearly["f2"],
    mode="lines+markers+text",
    line=dict(color=AZUL_CORP, width=3),
    marker=dict(size=12, color=AZUL_CORP, line=dict(width=2, color="white")),
    text=[f"{v:.3f}" for v in drift_yearly["f2"]],
    textposition="top center",
    textfont=dict(color=AZUL_CORP, size=12),
    name="F2",
    hovertemplate="Año %{x}<br>F2 = %{y:.4f}<br>Sprints: %{customdata:,}<extra></extra>",
    customdata=drift_yearly["n_sprints"],
))

# Linea de objetivo
fig_temp.add_hline(y=f2_min_objetivo, line_dash="dash", line_color=GRIS_FUERTE,
                   annotation_text=f"Objetivo mínimo: F2 = {f2_min_objetivo}",
                   annotation_position="right",
                   annotation_font_color=GRIS_FUERTE)

fig_temp.update_layout(
    xaxis=dict(title="Año del sprint", dtick=1),
    yaxis=dict(title="F2-score", range=[0.85, 1.0]),
    showlegend=False,
)
aplicar_estilo(fig_temp, alto=400)

col_g, col_n = st.columns([3, 2])

with col_g:
    st.plotly_chart(fig_temp, use_container_width=True)

with col_n:
    f2_inicio = drift_yearly.iloc[0]["f2"]
    f2_fin = drift_yearly.iloc[-1]["f2"]
    año_inicio = int(drift_yearly.iloc[0]["año"])
    año_fin = int(drift_yearly.iloc[-1]["año"])

    cambio = f2_fin - f2_inicio
    if abs(cambio) < 0.02:
        texto_cambio = f"se mantiene estable ({cambio:+.3f})"
    elif cambio < 0:
        texto_cambio = f"ha bajado {abs(cambio):.3f} puntos"
    else:
        texto_cambio = f"ha subido {cambio:.3f} puntos"

    st.markdown(
        f"<div style='background:#f9fafb; border-radius:8px; padding:14px 18px; "
        f"font-size:14px; color:#374151; line-height:1.6;'>"
        f"<b>Lectura</b><br><br>"
        f"Entre {año_inicio} y {año_fin}, el F2 del modelo {texto_cambio}.<br><br>"
        f"El año con menor desempeño es <b>{int(drift_yearly.loc[drift_yearly['f2'].idxmin(), 'año'])}</b> "
        f"con F2 = {drift_yearly['f2'].min():.3f}. "
        f"El año con mejor desempeño es <b>{int(drift_yearly.loc[drift_yearly['f2'].idxmax(), 'año'])}</b> "
        f"con F2 = {drift_yearly['f2'].max():.3f}.<br><br>"
        f"Las bandas verde, ámbar y roja indican zonas de desempeño aceptable, "
        f"de monitoreo y de atención respectivamente."
        f"</div>",
        unsafe_allow_html=True
    )


st.markdown(" ")
st.divider()


# ========================================================
# Drift por feature
# ========================================================
st.markdown("### Estabilidad de las variables (PSI)")
st.markdown(
    "<p style='color:#6b7280; font-size:14px;'>"
    "El PSI mide que tanto cambio la "
    "distribucion de cada variable entre los datos de entrenamiento "
    "(2000-2014) y los recientes. Si pasa de 0.25 hay que mirar."
    "</p>",
    unsafe_allow_html=True
)

drift_top = drift_psi.head(12).copy()
drift_top["feature_legible"] = drift_top["feature"].map(desc).fillna(drift_top["feature"])

def color_psi(p: float) -> str:
    if p > 0.25: return ROJO_ATENCION
    if p > 0.10: return AMBAR_MEDIO
    return VERDE_OK

colores_psi = [color_psi(p) for p in drift_top["psi"]]

fig_psi = go.Figure(go.Bar(
    x=drift_top["psi"][::-1],
    y=drift_top["feature_legible"][::-1],
    orientation="h",
    marker_color=colores_psi[::-1],
    text=[f"{v:.3f}" for v in drift_top["psi"][::-1]],
    textposition="outside",
    showlegend=False,
))

# Lineas de referencia
fig_psi.add_vline(x=0.10, line_dash="dot", line_color=AMBAR_MEDIO,
                  annotation_text="Monitorear (0.10)",
                  annotation_position="top")
fig_psi.add_vline(x=0.25, line_dash="dash", line_color=ROJO_ATENCION,
                  annotation_text="Revisar (0.25)",
                  annotation_position="top")

fig_psi.update_layout(
    xaxis_title="PSI (mayor = más cambio en la distribución)",
    yaxis=dict(autorange="reversed"),
    bargap=0.3,
)
aplicar_estilo(fig_psi, alto=460)


col_psi, col_acc = st.columns([3, 2])

with col_psi:
    st.plotly_chart(fig_psi, use_container_width=True)

with col_acc:
    st.markdown("**Acciones recomendadas**")

    accion_items = []
    if delta_f2 < -0.05:
        accion_items.append(
            ("🔴 Reentrenar modelo",
             "El F2 cayó más de 0.05 puntos respecto a la referencia. "
             "Vale la pena reentrenar con datos recientes.")
        )
    if n_drift_alto >= 3:
        accion_items.append(
            ("🔴 Revisar variables con PSI alto",
             f"Hay {n_drift_alto} variables con drift severo. "
             "Verificar si el cambio en sus distribuciones es estructural.")
        )
    if not accion_items:
        accion_items.append(
            ("🟢 Modelo en condiciones de operar",
             "El desempeño está dentro de los rangos esperados. "
             "Continuar con monitoreo periódico (sugerido: mensual).")
        )

    # Siempre sugerir recalibrar Brier periodicamente
    if metricas["brier"] > 0.10:
        accion_items.append(
            ("🟡 Recalibrar probabilidades",
             "El Brier score subió por encima de 0.10. "
             "Una recalibración con datos recientes ajustaría las probabilidades.")
        )

    for titulo, detalle in accion_items:
        st.markdown(
            f"<div style='background:#f9fafb; padding:12px 16px; border-radius:8px; "
            f"margin-bottom:8px;'>"
            f"<b style='font-size:14px;'>{titulo}</b><br>"
            f"<span style='color:#6b7280; font-size:13px;'>{detalle}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        f"<div style='background:#eff6ff; border-left:3px solid {AZUL_CORP}; "
        f"padding:10px 14px; border-radius:6px; margin-top:6px; "
        f"font-size:13px; color:#1e3a8a;'>"
        f"<b>Próxima revisión sugerida:</b> 30 días"
        f"</div>",
        unsafe_allow_html=True
    )
