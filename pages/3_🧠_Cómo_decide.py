"""Pagina: Como decide
Pregunta unica: ¿Por que confiar en este modelo para priorizar?
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import (cargar_shap_global, cargar_reliability,
                         cargar_metricas, feature_descripcion)
from utils.estilo import (aplicar_estilo, ROJO_ATENCION, AMBAR_MEDIO, VERDE_OK,
                          AZUL_CORP, GRIS_FUERTE, GRIS_MEDIO, GRIS_SUAVE)


from utils.theme import aplicar_tema, COLORS, nota_lateral, separador
aplicar_tema("Como decide · ShiftMetrics", "🧠")


st.markdown("## 🧠 Cómo decide el modelo")
st.markdown(
    "<p style='color:#6b7280; font-size:16px;'>"
    "Que variables pesan mas y por que las probabilidades "
    "que da el modelo son creibles para tomar decisiones."
    "</p>",
    unsafe_allow_html=True
)
st.divider()


shap_global = cargar_shap_global()
reliability = cargar_reliability()
metricas    = cargar_metricas()
desc        = feature_descripcion()


# ========================================================
# Bloque 1: Importancia global de variables
# ========================================================
st.markdown("### Qué variables pesan más para el modelo")

# Top 10 features con etiquetas legibles
top10 = shap_global.head(10).copy()
top10["feature_legible"] = top10["feature"].map(desc).fillna(top10["feature"])
top10["importancia_pct"] = top10["importancia"] / top10["importancia"].sum() * 100

# Colorear: top 3 en azul corporativo, resto en gris (jerarquia visual)
colores = [AZUL_CORP if i < 3 else GRIS_MEDIO for i in range(len(top10))]

fig_imp = go.Figure(go.Bar(
    x=top10["importancia_pct"][::-1],
    y=top10["feature_legible"][::-1],
    orientation="h",
    marker_color=colores[::-1],
    text=[f"{v:.1f}%" for v in top10["importancia_pct"][::-1]],
    textposition="outside",
    showlegend=False,
))
fig_imp.update_layout(
    xaxis_title="Peso relativo en las decisiones del modelo (%)",
    yaxis=dict(autorange="reversed"),
    bargap=0.25,
)
aplicar_estilo(fig_imp, alto=460)

col_imp, col_text = st.columns([3, 2])

with col_imp:
    st.plotly_chart(fig_imp, use_container_width=True)

with col_text:
    st.markdown("**Lectura**")
    st.markdown(
        "<div style='background:#f9fafb; border-radius:8px; padding:14px 18px; "
        "font-size:14px; color:#374151; line-height:1.6;'>"
        f"La variable con mayor peso es <b>{top10.iloc[0]['feature_legible']}</b>, "
        f"que concentra cerca del {top10.iloc[0]['importancia_pct']:.0f}% de las "
        "decisiones del modelo.<br><br>"
        f"Las tres variables principales explican aproximadamente el "
        f"{top10.head(3)['importancia_pct'].sum():.0f}% del comportamiento. "
        "El resto contribuye pero en menor medida."
        "</div>",
        unsafe_allow_html=True
    )

# Explicaciones de las top 3
st.markdown(" ")
st.markdown("**Por qué cada variable pesa tanto**")
for _, row in top10.head(5).iterrows():
    nombre_legible = desc.get(row["feature"], row["feature"])
    st.markdown(
        f"<div style='border-left:3px solid {AZUL_CORP}; padding:6px 14px; "
        f"margin:6px 0; background:#fafafa;'>"
        f"<b>{nombre_legible}</b> · {row['importancia_pct']:.1f}%<br>"
        f"<span style='color:#6b7280; font-size:14px;'>{row['explicacion']}</span>"
        f"</div>",
        unsafe_allow_html=True
    )


st.markdown(" ")
st.divider()


# ========================================================
# Bloque 2: Calibracion (que tan confiables son las probabilidades)
# ========================================================
st.markdown("### Las probabilidades son fieles a la realidad")
st.markdown(
    "<p style='color:#6b7280; font-size:14px;'>"
    "Cuando el modelo dice '30%', en la practica deberian "
    "haber bugs escapados en mas o menos 30 de cada 100 sprints parecidos. "
    "Esta curva verifica que eso se cumpla."
    "</p>",
    unsafe_allow_html=True
)

col_cal, col_brier = st.columns([3, 2])

with col_cal:
    fig_cal = go.Figure()

    # Linea ideal (diagonal)
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=GRIS_MEDIO, dash="dash", width=2),
        name="Calibración perfecta",
    ))

    # Curva real
    fig_cal.add_trace(go.Scatter(
        x=reliability["prob_predicha_media"],
        y=reliability["frecuencia_real"],
        mode="lines+markers",
        line=dict(color=AZUL_CORP, width=3),
        marker=dict(size=10, color=AZUL_CORP),
        name="Modelo en datos recientes",
        text=[f"n={int(n):,}" for n in reliability["n"]],
        hovertemplate=("Predicha media: %{x:.2f}<br>"
                       "Frecuencia real: %{y:.2f}<br>"
                       "%{text}<extra></extra>"),
    ))

    fig_cal.update_layout(
        xaxis=dict(title="Probabilidad predicha", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title="Frecuencia real observada", range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    aplicar_estilo(fig_cal, alto=440)
    st.plotly_chart(fig_cal, use_container_width=True)


with col_brier:
    st.markdown("**Indicadores de calibración**")

    brier = metricas["brier"]
    if brier < 0.08:
        brier_color, brier_emoji, brier_status = VERDE_OK, "🟢", "Excelente"
    elif brier < 0.12:
        brier_color, brier_emoji, brier_status = AMBAR_MEDIO, "🟡", "Aceptable"
    else:
        brier_color, brier_emoji, brier_status = ROJO_ATENCION, "🔴", "Requiere revisión"

    st.markdown(
        f"<div style='background:#ffffff; border:1px solid #e5e7eb; "
        f"border-radius:10px; padding:16px;'>"
        f"<div style='color:#6b7280; font-size:13px;'>Brier score</div>"
        f"<div style='font-size:32px; font-weight:800; color:{brier_color};'>"
        f"{brier:.4f}</div>"
        f"<div style='color:{brier_color}; font-size:14px; font-weight:600;'>"
        f"{brier_emoji} {brier_status}</div>"
        f"<div style='color:#6b7280; font-size:12px; margin-top:8px;'>"
        f"Mide el error medio cuadrado de la probabilidad. "
        f"Menor es mejor. La referencia es estar por debajo de 0.10."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(" ")
    st.markdown(
        "<div style='background:#f9fafb; padding:14px 18px; border-radius:8px; "
        "font-size:14px; color:#374151;'>"
        "<b>Cómo se logró la calibración</b><br><br>"
        "El modelo base se entrenó hasta 2014. Las probabilidades se calibraron "
        "con un conjunto independiente del año 2015 usando regresión isotónica. "
        "Esto separa el ajuste del modelo del ajuste de probabilidades, evitando "
        "sobreoptimismo."
        "</div>",
        unsafe_allow_html=True
    )


st.markdown(" ")
st.divider()


# ========================================================
# Bloque 3: Frente a un predictor trivial
# ========================================================
st.markdown("### Comparación con la línea base")
st.markdown(
    "<p style='color:#6b7280; font-size:14px;'>"
    "La forma mas facil de predecir bugs escapados es decir siempre si, "
    "aprovechando que la mayoria de sprints tienen algun bug. "
    "El modelo gana por <b>precisión</b>, no por capturar más alarmas."
    "</p>",
    unsafe_allow_html=True
)


# Comparativa: baseline vs modelo (F2 + precision + recall + Brier)
metricas_baseline = {
    "F2": 0.916,
    "Precisión": 0.704,    # tasa base de positivos en train
    "Recall": 1.000,
    "Brier": 0.217,
}
metricas_modelo = {
    "F2": metricas["f2"],
    "Precisión": metricas["precision"],
    "Recall": metricas["recall"],
    "Brier": 1 - metricas["brier"],   # invertir para que mas alto = mejor visualmente
}
# Para Brier, mostrar 1-brier para que todas las metricas vayan en la misma direccion
metricas_baseline_disp = {**metricas_baseline, "Brier (invertido)": 1 - metricas_baseline["Brier"]}
del metricas_baseline_disp["Brier"]
metricas_modelo_disp = {**metricas_modelo, "Brier (invertido)": metricas_modelo["Brier"]}
del metricas_modelo_disp["Brier"]

# Hacerlo simple: 3 metricas comparadas, omitir Brier para no confundir
nombres = ["F2", "Precisión", "Recall"]
v_base  = [metricas_baseline[k] for k in nombres]
v_model = [metricas["f2"], metricas["precision"], metricas["recall"]]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=nombres, y=v_base, name="Predictor trivial (siempre dice sí)",
    marker_color=GRIS_MEDIO,
    text=[f"{v:.3f}" for v in v_base], textposition="outside",
))
fig_comp.add_trace(go.Bar(
    x=nombres, y=v_model, name="Modelo champion",
    marker_color=AZUL_CORP,
    text=[f"{v:.3f}" for v in v_model], textposition="outside",
))
fig_comp.update_layout(
    barmode="group",
    yaxis=dict(range=[0, 1.15], title="Valor"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
aplicar_estilo(fig_comp, alto=460)

col_g, col_t = st.columns([3, 2])
with col_g:
    st.plotly_chart(fig_comp, use_container_width=True)
with col_t:
    st.markdown(
        f"<div style='background:#f9fafb; border-radius:8px; padding:14px 18px; "
        f"font-size:14px; color:#374151; line-height:1.6;'>"
        f"<b>Por qué importa esto</b><br><br>"
        f"El predictor trivial alcanza F2 de 0.916 sin aprender nada, sólo por "
        f"el desbalance del dataset. Cualquier modelo razonable debe superarlo.<br><br>"
        f"El modelo actual logra F2 de <b>{metricas['f2']:.3f}</b> manteniendo recall "
        f"alto (<b>{metricas['recall']:.3f}</b>) y elevando la precisión a "
        f"<b>{metricas['precision']:.3f}</b>. Esto significa que las alertas que "
        f"emite tienen señal real, no son simple ruido de mayoría."
        f"</div>",
        unsafe_allow_html=True
    )
