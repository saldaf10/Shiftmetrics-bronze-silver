"""app.py — ShiftMetrics dashboard

Streamlit multipagina. Cada pagina responde una sola pregunta
con una visualizacion principal y elementos minimos de apoyo.
"""
from __future__ import annotations

import streamlit as st

from utils.data import cargar_metricas


st.set_page_config(
    page_title="ShiftMetrics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS de afinado: tipografia, espaciados, eliminar lo que distrae
st.markdown("""
<style>
  /* Tipografia base */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  /* Header sin el menu de Streamlit */
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
  /* Tarjetas KPI */
  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  }
  [data-testid="stMetricLabel"] { color: #6b7280; font-size: 13px; font-weight: 600; }
  [data-testid="stMetricValue"] { color: #111827; font-weight: 700; }
  /* Sidebar */
  [data-testid="stSidebar"] { background: #f9fafb; }
  /* Botones */
  .stButton button {
    border-radius: 10px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: #111827;
    font-weight: 600;
  }
  .stButton button:hover { border-color: #1e3a8a; color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)


metricas = cargar_metricas()

# --- Hero ---
st.markdown(
    "<h1 style='margin-bottom:0'>ShiftMetrics</h1>"
    "<p style='color:#6b7280; font-size:18px; margin-top:4px;'>"
    "Anticipar qué sprints tienen alto riesgo de dejar escapar un defecto a producción."
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# --- Pregunta de oro ---
col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown("### La pregunta que resolvemos")
    st.markdown(
        "<div style='background:#f3f4f6; border-left:4px solid #1e3a8a; "
        "padding:18px 22px; border-radius:6px; font-size:17px; color:#111827;'>"
        "<b>¿En qué sprints conviene concentrar el esfuerzo de revisión antes "
        "del cierre para evitar que escape un defecto a producción?</b>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(" ")
    st.markdown(
        "Cada sprint recibe una probabilidad de defecto escapado entre 0 y 1. "
        "Sobre esa probabilidad se aplica un umbral operativo que separa "
        "alto riesgo, riesgo intermedio y riesgo bajo. El propósito es priorizar "
        "el trabajo de QA, no reemplazar el criterio del equipo."
    )

with col_b:
    st.markdown("### Estado del modelo")
    c1, c2 = st.columns(2)
    c1.metric("F2 en datos recientes", f"{metricas['f2']:.3f}",
              help="F2 da el doble de peso al recall que a la precisión. Es la métrica primaria.")
    c2.metric("Recall", f"{metricas['recall']:.3f}",
              help="Proporción de sprints con defecto que el modelo logra marcar.")
    c1.metric("Precisión", f"{metricas['precision']:.3f}",
              help="De las alertas que emite el modelo, qué proporción acertaron.")
    c2.metric("Brier score", f"{metricas['brier']:.3f}",
              help="Calidad de la calibración de probabilidades. Menor es mejor.")


st.divider()


# --- Como navegar ---
st.markdown("### Cómo recorrer el tablero")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        "**🚦 Riesgo del periodo**  \n"
        "Ranking de sprints ordenados por probabilidad. "
        "Punto de partida para la decisión semanal."
    )

with col2:
    st.markdown(
        "**🔬 Simulador**  \n"
        "Probar un sprint hipotético cambiando sus métricas. "
        "Útil para entender qué mueve el riesgo."
    )

with col3:
    st.markdown(
        "**🧠 Cómo decide**  \n"
        "Qué variables pesan más en general y por qué el modelo "
        "es confiable como herramienta de priorización."
    )

with col4:
    st.markdown(
        "**📡 Salud del modelo**  \n"
        "Si el desempeño se ha deteriorado año a año y si hay "
        "señales de que conviene reentrenar."
    )


st.markdown(" ")

st.info(
    "Las páginas se recorren en orden con el menú lateral, "
    "pero cada una se puede leer de forma independiente."
)


# --- Pie ---
st.markdown(" ")
st.markdown(
    "<div style='color:#9ca3af; font-size:13px; text-align:right;'>"
    f"Champion calibrado · {metricas.get('registry_name','')} {metricas.get('registry_version','')} · "
    f"Evaluación: {metricas.get('fecha_evaluacion','')}"
    "</div>",
    unsafe_allow_html=True
)
