"""Pagina: Simulador
Pregunta unica: ¿Como cambia el riesgo al variar las metricas de un sprint?
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import cargar_champion, cargar_metricas, feature_descripcion
from utils.estilo import (aplicar_estilo, ROJO_ATENCION, AMBAR_MEDIO, VERDE_OK,
                          GRIS_FUERTE, GRIS_MEDIO)


from utils.theme import aplicar_tema, COLORS, nota_lateral, separador
aplicar_tema("Simulador · ShiftMetrics", "🔬")


# --- Header ---
st.markdown("## 🔬 Simulador de sprint")
st.markdown(
    "<p style='color:#6b7280; font-size:16px;'>"
    "Mover los numeros de un sprint inventado y ver como cambia "
    "la probabilidad. Sirve para entender que mueve el riesgo."
    "</p>",
    unsafe_allow_html=True
)
st.divider()


# --- Cargar modelo ---
artefactos = cargar_champion()
modelo    = artefactos["model"]
features  = artefactos.get("feature_cols", [])
umbral    = artefactos.get("threshold", 0.220)

# Indicador de fuente del modelo
modelo_source = artefactos.get("source", "local")
if "mlflow" in str(modelo_source):
    st.sidebar.success(f"Modelo: MLflow ({modelo_source})")
else:
    st.sidebar.info(f"Modelo: {modelo_source}")
metricas  = cargar_metricas()
desc      = feature_descripcion()


# --- Modo basico vs avanzado ---
modo = st.radio(
    "Nivel de detalle",
    ["Modo básico (5 variables principales)", "Modo avanzado (todas las variables)"],
    horizontal=True,
    help="El basico calcula las derivadas solo. "
         "El avanzado deja mover todo a mano."
)


st.markdown(" ")

col_inputs, col_output = st.columns([3, 2])


# --- Inputs ---
with col_inputs:
    st.markdown("### Configuración del sprint")

    if modo.startswith("Modo básico"):
        col1, col2 = st.columns(2)
        num_bugs = col1.number_input(
            "Bugs en el sprint", min_value=0, max_value=100, value=5, step=1,
            help="Cuantos bugs tiene este sprint."
        )
        num_stories = col2.number_input(
            "Stories en el sprint", min_value=0, max_value=100, value=8, step=1
        )
        num_tasks = col1.number_input(
            "Tasks en el sprint", min_value=0, max_value=100, value=10, step=1
        )
        cycle_days = col2.number_input(
            "Tiempo de ciclo promedio (días)", min_value=0.0, max_value=2000.0,
            value=45.0, step=5.0,
            help="Promedio de dias que tarda un issue en cerrarse."
        )

        col3, col4 = st.columns(2)
        año = col3.number_input("Año del sprint", min_value=2000, max_value=2025,
                                value=2021, step=1)
        mes = col4.slider("Mes del sprint", 1, 12, 6)

        # Derivar automaticamente
        total = max(num_bugs + num_stories + num_tasks, 1)
        bsr = num_bugs / max(num_stories, 1) if num_stories > 0 else None

        inputs = {
            "num_bugs_sprint": float(num_bugs),
            "num_stories_sprint": float(num_stories),
            "num_tasks_sprint": float(num_tasks),
            "total_issues_sprint": float(total),
            "log_avg_cycle_time": math.log1p(cycle_days),
            "log_bug_story_ratio": math.log1p(bsr) if bsr is not None else 0.0,
            "log_total_issues": math.log1p(total),
            "sprint_year": float(año),
            "sprint_month_sin": math.sin(2 * math.pi * mes / 12),
            "sprint_month_cos": math.cos(2 * math.pi * mes / 12),
            "deploy_frequency_weekly": 0.0,
            "change_failure_rate": 0.0,
            "bsr_missing": int(bsr is None),
            "cycle_missing": int(cycle_days == 0),
            "dora_missing": 1,
            "bugs_per_issue": num_bugs / total,
            "log_cycle_x_bsr": math.log1p(cycle_days) * (math.log1p(bsr) if bsr is not None else 0.0),
        }

        with st.expander("Cómo se calculan las variables derivadas"):
            st.markdown(
                f"- **Densidad de bugs** = bugs / total issues = {num_bugs}/{total} = "
                f"{num_bugs/total:.3f}\n"
                f"- **Ratio bugs/stories** = "
                f"{'sin stories, se marca como faltante' if num_stories==0 else f'{num_bugs}/{num_stories} = {num_bugs/num_stories:.2f}'}\n"
                f"- **Mes cíclico** = encoding seno/coseno del mes para preservar "
                f"continuidad entre diciembre y enero\n"
                f"- **Métricas DORA**: no se proveen en modo básico (cobertura típica < 10%)"
            )

    else:
        # Modo avanzado: cada feature se controla
        col1, col2 = st.columns(2)
        num_bugs = col1.number_input("Bugs", min_value=0, max_value=200, value=5)
        num_stories = col2.number_input("Stories", min_value=0, max_value=200, value=8)
        num_tasks = col1.number_input("Tasks", min_value=0, max_value=200, value=10)
        cycle_days = col2.number_input("Tiempo de ciclo (días)",
                                        min_value=0.0, max_value=3000.0, value=45.0)

        col3, col4 = st.columns(2)
        año = col3.number_input("Año", min_value=2000, max_value=2025, value=2021)
        mes = col4.slider("Mes", 1, 12, 6)

        col5, col6 = st.columns(2)
        deploy_freq = col5.number_input(
            "Despliegues por semana",
            min_value=0.0, max_value=20.0, value=0.0, step=0.1,
            help="Metrica DORA. Si no la tienes, dejala en 0."
        )
        cfr = col6.number_input(
            "Tasa de cambios fallidos", min_value=0.0, max_value=1.0,
            value=0.0, step=0.05,
            help="Métrica DORA. 0.30 significa que 30% de los cambios fallan."
        )

        marcar_dora_missing = st.checkbox(
            "Marcar métricas DORA como faltantes (sprint sin instrumentación)",
            value=True,
            help="En el dataset real solo 9% de los sprints tiene DORA. "
                 "Si tu sprint no tiene, deja esta opción marcada."
        )

        marcar_bsr_missing = st.checkbox(
            "Marcar ratio bugs/stories como faltante",
            value=(num_stories == 0)
        )

        marcar_cycle_missing = st.checkbox(
            "Marcar tiempo de ciclo como faltante",
            value=(cycle_days == 0)
        )

        total = max(num_bugs + num_stories + num_tasks, 1)
        bsr = num_bugs / max(num_stories, 1) if num_stories > 0 else 0

        inputs = {
            "num_bugs_sprint": float(num_bugs),
            "num_stories_sprint": float(num_stories),
            "num_tasks_sprint": float(num_tasks),
            "total_issues_sprint": float(total),
            "log_avg_cycle_time": math.log1p(cycle_days) if not marcar_cycle_missing else 0.0,
            "log_bug_story_ratio": math.log1p(bsr) if not marcar_bsr_missing else 0.0,
            "log_total_issues": math.log1p(total),
            "sprint_year": float(año),
            "sprint_month_sin": math.sin(2 * math.pi * mes / 12),
            "sprint_month_cos": math.cos(2 * math.pi * mes / 12),
            "deploy_frequency_weekly": float(deploy_freq) if not marcar_dora_missing else 0.0,
            "change_failure_rate": float(cfr) if not marcar_dora_missing else 0.0,
            "bsr_missing": int(marcar_bsr_missing),
            "cycle_missing": int(marcar_cycle_missing),
            "dora_missing": int(marcar_dora_missing),
            "bugs_per_issue": num_bugs / total,
            "log_cycle_x_bsr": (math.log1p(cycle_days) if not marcar_cycle_missing else 0.0) *
                                (math.log1p(bsr) if not marcar_bsr_missing else 0.0),
        }


# --- Prediccion ---
X = np.array([[inputs[f] for f in features]])
prob = float(modelo.predict_proba(X)[0, 1])

if prob >= 0.50:
    categoria, color, emoji = "alto", ROJO_ATENCION, "🔴"
elif prob >= umbral:
    categoria, color, emoji = "medio", AMBAR_MEDIO, "🟡"
else:
    categoria, color, emoji = "bajo", VERDE_OK, "🟢"


# --- Output ---
with col_output:
    st.markdown("### Resultado")

    # Tarjeta resumen
    st.markdown(
        f"<div style='background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; "
        f"padding:24px; text-align:center;'>"
        f"<div style='color:#6b7280; font-size:13px;'>Probabilidad estimada</div>"
        f"<div style='font-size:54px; font-weight:800; color:{color}; line-height:1;'>"
        f"{prob*100:.1f}%</div>"
        f"<div style='font-size:18px; color:{color}; font-weight:600; margin-top:6px;'>"
        f"{emoji} Riesgo {categoria}</div>"
        f"<div style='color:#6b7280; font-size:13px; margin-top:14px;'>"
        f"Umbral operativo: <b>{umbral:.3f}</b></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Barra visual de la probabilidad sobre el espectro
    fig_bar = go.Figure()
    # Banda verde 0-22, ambar 22-50, roja 50-100
    fig_bar.add_shape(type="rect", x0=0, x1=umbral, y0=0, y1=1,
                      fillcolor=VERDE_OK, opacity=0.25, line_width=0)
    fig_bar.add_shape(type="rect", x0=umbral, x1=0.5, y0=0, y1=1,
                      fillcolor=AMBAR_MEDIO, opacity=0.25, line_width=0)
    fig_bar.add_shape(type="rect", x0=0.5, x1=1.0, y0=0, y1=1,
                      fillcolor=ROJO_ATENCION, opacity=0.25, line_width=0)
    # Marca de la probabilidad actual
    fig_bar.add_shape(type="line", x0=prob, x1=prob, y0=0, y1=1,
                      line=dict(color=color, width=4))
    fig_bar.add_annotation(x=prob, y=1.15, text=f"{prob*100:.1f}%",
                           showarrow=False, font=dict(color=color, size=14, family="Inter"))

    fig_bar.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%",
                   tickvals=[0, umbral, 0.5, 1.0],
                   ticktext=["0%", f"{umbral*100:.0f}%", "50%", "100%"]),
        yaxis=dict(range=[0, 1], visible=False),
        height=110,
        margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# --- Sensibilidad: como cambia el riesgo al variar cada feature ---
st.markdown(" ")
st.divider()
st.markdown("### Análisis de sensibilidad")
st.markdown(
    "<p style='color:#6b7280; font-size:14px;'>"
    "Como cambiaria la probabilidad si movemos una variable dejando "
    "el resto fijo. Ayuda a ver cuales son las palancas."
    "</p>",
    unsafe_allow_html=True
)

# Variables sobre las cuales hacer barrido
opciones_barrido = {
    "Bugs en el sprint": "num_bugs_sprint",
    "Tiempo de ciclo (días)": "avg_cycle_time_days_raw",
    "Stories en el sprint": "num_stories_sprint",
    "Tasks en el sprint": "num_tasks_sprint",
}
var_legible = st.selectbox("Variable a explorar", list(opciones_barrido.keys()))
var_key = opciones_barrido[var_legible]


# Rangos sensibles
if var_key == "num_bugs_sprint":
    valores = np.arange(0, 31, 1)
elif var_key == "avg_cycle_time_days_raw":
    valores = np.arange(0, 301, 10)
elif var_key == "num_stories_sprint":
    valores = np.arange(0, 31, 1)
else:
    valores = np.arange(0, 31, 1)


probs = []
for v in valores:
    inp = inputs.copy()
    if var_key == "num_bugs_sprint":
        inp["num_bugs_sprint"] = float(v)
        inp["total_issues_sprint"] = max(v + inp["num_stories_sprint"] + inp["num_tasks_sprint"], 1)
        inp["log_total_issues"] = math.log1p(inp["total_issues_sprint"])
        inp["bugs_per_issue"] = v / inp["total_issues_sprint"]
        if inp["num_stories_sprint"] > 0:
            bsr_nuevo = v / inp["num_stories_sprint"]
            inp["log_bug_story_ratio"] = math.log1p(bsr_nuevo)
            inp["log_cycle_x_bsr"] = inp["log_avg_cycle_time"] * inp["log_bug_story_ratio"]
    elif var_key == "avg_cycle_time_days_raw":
        inp["log_avg_cycle_time"] = math.log1p(float(v))
        inp["cycle_missing"] = int(v == 0)
        inp["log_cycle_x_bsr"] = inp["log_avg_cycle_time"] * inp["log_bug_story_ratio"]
    elif var_key == "num_stories_sprint":
        inp["num_stories_sprint"] = float(v)
        inp["total_issues_sprint"] = max(inp["num_bugs_sprint"] + v + inp["num_tasks_sprint"], 1)
        inp["log_total_issues"] = math.log1p(inp["total_issues_sprint"])
        inp["bugs_per_issue"] = inp["num_bugs_sprint"] / inp["total_issues_sprint"]
        if v > 0:
            bsr_nuevo = inp["num_bugs_sprint"] / v
            inp["log_bug_story_ratio"] = math.log1p(bsr_nuevo)
            inp["log_cycle_x_bsr"] = inp["log_avg_cycle_time"] * inp["log_bug_story_ratio"]
            inp["bsr_missing"] = 0
        else:
            inp["log_bug_story_ratio"] = 0
            inp["log_cycle_x_bsr"] = 0
            inp["bsr_missing"] = 1
    elif var_key == "num_tasks_sprint":
        inp["num_tasks_sprint"] = float(v)
        inp["total_issues_sprint"] = max(inp["num_bugs_sprint"] + inp["num_stories_sprint"] + v, 1)
        inp["log_total_issues"] = math.log1p(inp["total_issues_sprint"])
        inp["bugs_per_issue"] = inp["num_bugs_sprint"] / inp["total_issues_sprint"]

    X_test = np.array([[inp[f] for f in features]])
    probs.append(float(modelo.predict_proba(X_test)[0, 1]))

fig_sens = go.Figure()
# Bandas semaforo de fondo
fig_sens.add_hrect(y0=0, y1=umbral, fillcolor=VERDE_OK, opacity=0.10, line_width=0)
fig_sens.add_hrect(y0=umbral, y1=0.50, fillcolor=AMBAR_MEDIO, opacity=0.10, line_width=0)
fig_sens.add_hrect(y0=0.50, y1=1.0, fillcolor=ROJO_ATENCION, opacity=0.10, line_width=0)
# Linea de probabilidad
fig_sens.add_trace(go.Scatter(
    x=valores, y=probs, mode="lines+markers",
    line=dict(color=GRIS_FUERTE, width=3),
    marker=dict(size=6, color=GRIS_FUERTE),
    showlegend=False
))
# Umbral
fig_sens.add_hline(y=umbral, line_dash="dash", line_color=GRIS_MEDIO,
                   annotation_text=f"Umbral {umbral:.3f}",
                   annotation_position="right")

fig_sens.update_layout(
    xaxis_title=var_legible,
    yaxis_title="Probabilidad de defecto escapado",
    yaxis=dict(range=[0, 1], tickformat=".0%"),
)
aplicar_estilo(fig_sens, alto=440)
st.plotly_chart(fig_sens, use_container_width=True)


# --- Interpretacion automatica ---
delta = probs[-1] - probs[0]
if abs(delta) < 0.05:
    interp = "tiene un efecto modesto sobre la probabilidad estimada"
    color_int = GRIS_MEDIO
elif delta > 0:
    interp = f"empuja el riesgo hacia arriba: pasa de {probs[0]*100:.1f}% a {probs[-1]*100:.1f}%"
    color_int = ROJO_ATENCION
else:
    interp = f"reduce el riesgo: pasa de {probs[0]*100:.1f}% a {probs[-1]*100:.1f}%"
    color_int = VERDE_OK

st.markdown(
    f"<div style='background:#f9fafb; border-left:4px solid {color_int}; "
    f"padding:12px 18px; border-radius:6px; color:#111827;'>"
    f"<b>Lectura:</b> al variar <b>{var_legible}</b> en el rango "
    f"[{valores[0]}, {valores[-1]}], {interp} para este sprint hipotético."
    f"</div>",
    unsafe_allow_html=True
)
