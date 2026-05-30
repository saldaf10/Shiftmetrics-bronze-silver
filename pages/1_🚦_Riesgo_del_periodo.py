"""Ranking de sprints por riesgo."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.data import cargar_snapshot, cargar_metricas, feature_descripcion
from utils.estilo import (aplicar_estilo, color_riesgo, emoji_riesgo,
                          ROJO_ATENCION, AMBAR_MEDIO, VERDE_OK, GRIS_MEDIO)

from utils.theme import aplicar_tema, COLORS, nota_lateral, separador
aplicar_tema("Riesgo · ShiftMetrics", "🚦")

st.markdown("## 🚦 Riesgo del periodo")
st.markdown(
    "Sprints ordenados de mayor a menor probabilidad. "
    "Los de arriba son los que necesitan atencion primero."
)
st.divider()

df = cargar_snapshot()
met = cargar_metricas()
desc = feature_descripcion()

with st.sidebar:
    st.markdown("### Filtros")
    años = sorted(df["sprint_year"].unique().tolist())
    año_sel = st.multiselect("Año", años, default=años)
    proyectos = sorted(df["project"].unique().tolist())
    proy_sel = st.multiselect("Proyecto", proyectos, default=[])
    st.caption(f"Umbral operativo: {met['threshold']:.3f}")

df_f = df[df["sprint_year"].isin(año_sel)] if año_sel else df
if proy_sel:
    df_f = df_f[df_f["project"].isin(proy_sel)]

n_total = len(df_f)
n_alto  = (df_f["riesgo_categoria"] == "alto").sum()
n_medio = (df_f["riesgo_categoria"] == "medio").sum()
n_bajo  = (df_f["riesgo_categoria"] == "bajo").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sprints", f"{n_total:,}")
c2.metric("🔴 Alto", f"{n_alto:,}", f"{n_alto/max(n_total,1)*100:.1f}%")
c3.metric("🟡 Medio", f"{n_medio:,}", f"{n_medio/max(n_total,1)*100:.1f}%")
c4.metric("🟢 Bajo", f"{n_bajo:,}", f"{n_bajo/max(n_total,1)*100:.1f}%")

st.markdown(" ")
st.markdown("### Distribucion de probabilidades")

fig = go.Figure()
for color, lo, hi, label in [
    (VERDE_OK, 0.00, 0.22, "Bajo"), (AMBAR_MEDIO, 0.22, 0.50, "Medio"),
    (ROJO_ATENCION, 0.50, 1.01, "Alto")]:
    vals = df_f["probabilidad"][(df_f["probabilidad"]>=lo)&(df_f["probabilidad"]<hi)]
    fig.add_trace(go.Histogram(x=vals, xbins=dict(start=lo,end=hi,size=0.025),
                               marker_color=color, opacity=0.85, name=label))
fig.add_vline(x=met['threshold'], line_dash="dash", line_color="#374151",
              annotation_text=f"Umbral {met['threshold']:.3f}")
fig.update_layout(barmode="overlay", xaxis_title="Probabilidad", yaxis_title="Sprints",
                  legend=dict(orientation="h", y=1.02))
aplicar_estilo(fig, alto=380)
st.plotly_chart(fig, use_container_width=True)
st.caption("A la derecha del umbral estan los sprints que conviene revisar.")

st.markdown(" ")
st.markdown("### Ranking para revision")

top_n = st.slider("Cuantos mostrar", 10, 200, 25, 5)
ranking = df_f.sort_values("probabilidad", ascending=False).head(top_n).copy()
ranking["🚦"] = ranking["riesgo_categoria"].map(emoji_riesgo)
ranking["prob_%"] = (ranking["probabilidad"]*100).round(1)
ranking["driver"] = ranking["driver_1"].map(desc).fillna(ranking["driver_1"])
ranking["driver_2_leg"] = ranking["driver_2"].map(desc).fillna(ranking["driver_2"])

tabla = ranking[["🚦","sprint_id","project","sprint_year","prob_%",
                 "num_bugs_sprint","avg_cycle_time_days","driver","driver_2_leg"]].rename(columns={
    "sprint_id":"Sprint","project":"Proyecto","sprint_year":"Año",
    "prob_%":"Prob (%)","num_bugs_sprint":"Bugs","avg_cycle_time_days":"Ciclo (d)",
    "driver":"Driver 1","driver_2_leg":"Driver 2"})

st.dataframe(tabla, use_container_width=True, hide_index=True,
             column_config={"Prob (%)": st.column_config.ProgressColumn(
                 "Prob (%)", format="%.1f%%", min_value=0, max_value=100),
                 "Ciclo (d)": st.column_config.NumberColumn(format="%.0f"),
                 "Bugs": st.column_config.NumberColumn(format="%d")},
             height=min(420, 40+35*len(tabla)))

st.markdown(" ")
st.markdown("### Detalle de un sprint")

opciones = ranking["sprint_id"].tolist()
if opciones:
    sprint_sel = st.selectbox("Sprint", opciones, index=0)
    row = ranking[ranking["sprint_id"]==sprint_sel].iloc[0]

    ca, cb = st.columns([2, 3])
    with ca:
        cat_riesgo = row['riesgo_categoria']
        col_r = color_riesgo(cat_riesgo)
        st.markdown(
            f"<div style='background:#f9fafb;border-radius:10px;padding:18px;'>"
            f"<div style='font-size:22px;font-weight:700'>{row['sprint_id']}</div>"
            f"<div style='font-size:34px;font-weight:800;color:{col_r};'>"
            f"{row['probabilidad']*100:.1f}%</div>"
            f"<div>{emoji_riesgo(cat_riesgo)} Riesgo {cat_riesgo}</div>"
            f"<div style='margin-top:10px;color:#6b7280'>{row['project']} · {int(row['sprint_year'])}</div>"
            f"</div>", unsafe_allow_html=True)

    with cb:
        drv = pd.DataFrame({
            "variable": [desc.get(row["driver_1"],row["driver_1"]),
                         desc.get(row["driver_2"],row["driver_2"]),
                         desc.get(row["driver_3"],row["driver_3"])],
            "valor": [row["driver_1_val"], row["driver_2_val"], row["driver_3_val"]]})
        fig_d = go.Figure(go.Bar(x=drv["valor"], y=drv["variable"], orientation="h",
                                  marker_color=[ROJO_ATENCION if v>0 else VERDE_OK for v in drv["valor"]],
                                  text=[f"{v:+.2f}" for v in drv["valor"]], textposition="outside"))
        fig_d.update_layout(title="Que empuja el riesgo", xaxis_title="Contribucion",
                            yaxis=dict(autorange="reversed"), showlegend=False)
        aplicar_estilo(fig_d, alto=320)
        st.plotly_chart(fig_d, use_container_width=True)

    bugs_txt = int(row['num_bugs_sprint'])
    ciclo_txt = int(row['avg_cycle_time_days']) if pd.notna(row['avg_cycle_time_days']) else 'sin dato'
    st.markdown(
        f"<div style='background:#fefce8;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:6px;'>"
        f"Este sprint tiene {bugs_txt} bugs y un ciclo de {ciclo_txt} dias. "
        f"Lo que mas pesa es <b>{desc.get(row['driver_1'],row['driver_1'])}</b> "
        f"seguido de <b>{desc.get(row['driver_2'],row['driver_2'])}</b>."
        f"</div>", unsafe_allow_html=True)

st.markdown(" ")
csv = ranking.to_csv(index=False)
st.download_button("Descargar ranking CSV", csv, f"ranking_top{top_n}.csv", "text/csv")
