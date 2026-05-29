"""Pagina: Riesgo del periodo
Pregunta unica: ¿En que sprints conviene concentrar el esfuerzo de revision?
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data import cargar_snapshot, cargar_metricas, feature_descripcion
from utils.estilo import (aplicar_estilo, color_riesgo, emoji_riesgo,
                          ROJO_ATENCION, AMBAR_MEDIO, VERDE_OK, GRIS_MEDIO)


st.set_page_config(page_title="Riesgo del periodo · ShiftMetrics",
                   page_icon="🚦", layout="wide")


# --- Header ---
st.markdown("## 🚦 Riesgo del periodo")
st.markdown(
    "<p style='color:#6b7280; font-size:16px;'>"
    "Los sprints aparecen ordenados de mayor a menor probabilidad de dejar escapar "
    "un defecto. Las decisiones de revisión empiezan por arriba."
    "</p>",
    unsafe_allow_html=True
)
st.divider()


# --- Cargar ---
df = cargar_snapshot()
metricas = cargar_metricas()
desc = feature_descripcion()


# --- Filtros laterales: minimos ---
with st.sidebar:
    st.markdown("### Filtros")
    años = sorted(df["sprint_year"].unique().tolist())
    año_sel = st.multiselect("Año", años, default=años)
    proyectos = sorted(df["project"].unique().tolist())
    proy_sel = st.multiselect("Proyecto (opcional)", proyectos, default=[])

    st.markdown(" ")
    st.markdown(
        "<div style='color:#9ca3af; font-size:12px; padding:8px 0;'>"
        f"Umbral operativo: <b>{metricas['threshold']:.3f}</b><br>"
        "Por encima del umbral, el sprint se marca como alerta."
        "</div>",
        unsafe_allow_html=True
    )


# --- Aplicar filtros ---
df_f = df[df["sprint_year"].isin(año_sel)] if año_sel else df
if proy_sel:
    df_f = df_f[df_f["project"].isin(proy_sel)]


# --- Resumen de la pantalla ---
n_total = len(df_f)
n_alto  = (df_f["riesgo_categoria"] == "alto").sum()
n_medio = (df_f["riesgo_categoria"] == "medio").sum()
n_bajo  = (df_f["riesgo_categoria"] == "bajo").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sprints en el periodo", f"{n_total:,}")
col2.metric("Riesgo alto 🔴",  f"{n_alto:,}",   f"{n_alto/n_total*100:.1f}% del total")
col3.metric("Riesgo medio 🟡", f"{n_medio:,}",  f"{n_medio/n_total*100:.1f}% del total")
col4.metric("Riesgo bajo 🟢",  f"{n_bajo:,}",   f"{n_bajo/n_total*100:.1f}% del total")


st.markdown(" ")


# --- Visualizacion central: distribucion de probabilidades ---
st.markdown("### Cómo se reparte el riesgo")

# Histograma con colores semaforo (un solo grafico, claro)
hist_data = df_f["probabilidad"].values
fig_hist = go.Figure()

bins_low  = [b for b in range(0, 23, 2)]     # 0 a 0.22
bins_mid  = [b for b in range(22, 51, 2)]    # 0.22 a 0.50
bins_high = [b for b in range(50, 102, 2)]   # 0.50 a 1.00

import numpy as np

for color, prob_min, prob_max, label in [
    (VERDE_OK,      0.00, 0.22, "Riesgo bajo"),
    (AMBAR_MEDIO,   0.22, 0.50, "Riesgo medio"),
    (ROJO_ATENCION, 0.50, 1.01, "Riesgo alto"),
]:
    valores = hist_data[(hist_data >= prob_min) & (hist_data < prob_max)]
    fig_hist.add_trace(go.Histogram(
        x=valores, xbins=dict(start=prob_min, end=prob_max, size=0.025),
        marker_color=color, opacity=0.85, name=label,
    ))

fig_hist.add_vline(x=metricas['threshold'], line_dash="dash", line_color="#374151",
                   annotation_text=f"Umbral: {metricas['threshold']:.3f}",
                   annotation_position="top")
fig_hist.update_layout(
    barmode="overlay",
    xaxis_title="Probabilidad estimada de defecto escapado",
    yaxis_title="Cantidad de sprints",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
aplicar_estilo(fig_hist, alto=330)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(
    "<p style='color:#6b7280; font-size:14px; margin-top:-8px;'>"
    "La línea punteada marca el umbral operativo. Los sprints a su derecha "
    "son los que requieren revisión prioritaria."
    "</p>",
    unsafe_allow_html=True
)

st.markdown(" ")


# --- Ranking accionable ---
st.markdown("### Sprints priorizados para revisión")

top_n = st.slider("Mostrar los primeros", min_value=10, max_value=200, value=25, step=5)

ranking = df_f.sort_values("probabilidad", ascending=False).head(top_n).copy()
ranking["semaforo"] = ranking["riesgo_categoria"].map(emoji_riesgo)
ranking["probabilidad_pct"] = (ranking["probabilidad"] * 100).round(1)
ranking["driver_legible"] = ranking["driver_1"].map(desc).fillna(ranking["driver_1"])
ranking["driver_2_legible"] = ranking["driver_2"].map(desc).fillna(ranking["driver_2"])

# Tabla con styling
tabla = ranking[[
    "semaforo", "sprint_id", "project", "sprint_year",
    "probabilidad_pct", "num_bugs_sprint", "avg_cycle_time_days",
    "driver_legible", "driver_2_legible",
]].rename(columns={
    "semaforo": "",
    "sprint_id": "Sprint",
    "project": "Proyecto",
    "sprint_year": "Año",
    "probabilidad_pct": "Probabilidad (%)",
    "num_bugs_sprint": "Bugs",
    "avg_cycle_time_days": "Ciclo (días)",
    "driver_legible": "Driver principal",
    "driver_2_legible": "Driver secundario",
})

st.dataframe(
    tabla,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Probabilidad (%)": st.column_config.ProgressColumn(
            "Probabilidad (%)",
            format="%.1f%%",
            min_value=0, max_value=100,
        ),
        "Ciclo (días)": st.column_config.NumberColumn(format="%.0f"),
        "Bugs": st.column_config.NumberColumn(format="%d"),
    },
    height=min(420, 40 + 35 * len(tabla))
)


st.markdown(" ")


# --- Selector para inspeccion individual ---
st.markdown("### Inspección de un sprint específico")
st.markdown(
    "<p style='color:#6b7280; font-size:14px;'>"
    "Seleccione un sprint del ranking para ver qué variables están empujando el riesgo."
    "</p>",
    unsafe_allow_html=True
)

opciones = ranking["sprint_id"].tolist()
if opciones:
    sprint_elegido = st.selectbox("Sprint", opciones, index=0)
    row = ranking[ranking["sprint_id"] == sprint_elegido].iloc[0]

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown(
            f"<div style='background:#f9fafb; border-radius:10px; padding:18px;'>"
            f"<div style='color:#6b7280; font-size:13px;'>Sprint</div>"
            f"<div style='font-size:22px; font-weight:700; color:#111827;'>{row['sprint_id']}</div>"
            f"<div style='margin-top:14px; color:#6b7280; font-size:13px;'>Probabilidad</div>"
            f"<div style='font-size:34px; font-weight:800; color:{color_riesgo(row['riesgo_categoria'])};'>"
            f"{row['probabilidad']*100:.1f}%</div>"
            f"<div style='color:#6b7280; font-size:14px; margin-top:4px;'>"
            f"{emoji_riesgo(row['riesgo_categoria'])} Riesgo {row['riesgo_categoria']}</div>"
            f"<div style='margin-top:14px; color:#6b7280; font-size:13px;'>Proyecto · año</div>"
            f"<div style='font-size:16px; font-weight:600;'>{row['project']} · {int(row['sprint_year'])}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_b:
        # Waterfall de los 3 drivers principales
        drivers_data = pd.DataFrame({
            "variable": [
                desc.get(row["driver_1"], row["driver_1"]),
                desc.get(row["driver_2"], row["driver_2"]),
                desc.get(row["driver_3"], row["driver_3"]),
            ],
            "valor": [row["driver_1_val"], row["driver_2_val"], row["driver_3_val"]],
        })

        fig_drv = go.Figure(go.Bar(
            x=drivers_data["valor"],
            y=drivers_data["variable"],
            orientation="h",
            marker_color=[ROJO_ATENCION if v > 0 else VERDE_OK
                          for v in drivers_data["valor"]],
            text=[f"{v:+.2f}" for v in drivers_data["valor"]],
            textposition="outside",
        ))
        fig_drv.update_layout(
            title="Qué empuja el riesgo en este sprint",
            xaxis_title="Contribución relativa (positivo eleva riesgo)",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        aplicar_estilo(fig_drv, alto=280)
        st.plotly_chart(fig_drv, use_container_width=True)

    # Narrativa generada
    st.markdown(
        f"<div style='background:#fefce8; border-left:4px solid #f59e0b; "
        f"padding:14px 18px; border-radius:6px; color:#111827; margin-top:8px;'>"
        f"<b>Lectura:</b> este sprint tiene {int(row['num_bugs_sprint'])} bugs y un tiempo "
        f"de ciclo promedio de "
        f"{int(row['avg_cycle_time_days']) if pd.notna(row['avg_cycle_time_days']) else 'sin dato'} "
        f"días. Las variables que más empujan su clasificación son "
        f"<b>{desc.get(row['driver_1'], row['driver_1'])}</b> y "
        f"<b>{desc.get(row['driver_2'], row['driver_2'])}</b>. "
        f"Una revisión enfocada en esos drivers tiene mayor probabilidad de "
        f"prevenir el escape."
        f"</div>",
        unsafe_allow_html=True
    )


# --- Descarga ---
st.markdown(" ")
csv = ranking.drop(columns=["driver_legible", "driver_2_legible"]).to_csv(index=False)
st.download_button(
    "Descargar ranking en CSV",
    data=csv,
    file_name=f"ranking_top{top_n}.csv",
    mime="text/csv",
)
