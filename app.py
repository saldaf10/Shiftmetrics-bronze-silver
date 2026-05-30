"""ShiftMetrics — pagina de inicio."""
from __future__ import annotations
import streamlit as st
from utils.data import cargar_metricas

st.set_page_config(page_title="ShiftMetrics", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
  [data-testid="stMetric"] {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  }
  [data-testid="stSidebar"] { background: #f9fafb; }
</style>
""", unsafe_allow_html=True)

met = cargar_metricas()

st.markdown("<h1 style='margin-bottom:0'>ShiftMetrics</h1>", unsafe_allow_html=True)
st.markdown("Prediccion de defectos escapados en sprints del ecosistema Apache.")
st.divider()

col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown("### Que resuelve esto")
    st.markdown(
        "<div style='background:#f3f4f6; border-left:4px solid #1e3a8a; "
        "padding:18px 22px; border-radius:6px; font-size:17px; color:#111827;'>"
        "<b>¿En cuales sprints vale la pena reforzar la revision de calidad "
        "antes del cierre, para que no se escape un bug a produccion?</b>"
        "</div>", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        "El modelo le asigna a cada sprint una probabilidad de defecto escapado. "
        "Si esa probabilidad pasa del umbral (0.220), el sprint se marca para revision. "
        "No reemplaza el criterio del equipo — lo que hace es ordenar la cola de trabajo "
        "para que QA no pierda tiempo revisando sprints que probablemente estan bien."
    )

with col_b:
    st.markdown("### Numeros del modelo")
    c1, c2 = st.columns(2)
    c1.metric("F2", f"{met['f2']:.3f}",
              help="Metrica principal. Penaliza mas dejar pasar un bug que dar una alerta falsa.")
    c2.metric("Recall", f"{met['recall']:.3f}",
              help="De los sprints que si tenian bug, cuantos logro marcar.")
    c1.metric("Precision", f"{met['precision']:.3f}",
              help="De los que marco como riesgo, cuantos realmente tenian bug.")
    c2.metric("Brier", f"{met['brier']:.3f}",
              help="Error de calibracion. Mas bajo = las probabilidades son mas creibles.")

st.divider()

st.markdown("### Que hay en cada seccion")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("**📊 Exploración**\n\nGraficos de los EDAs iniciales sobre las 4 fuentes de datos del proyecto.")
with col2:
    st.markdown("**🚦 Riesgo**\n\nRanking de sprints de mayor a menor probabilidad. Aca se toman las decisiones.")
with col3:
    st.markdown("**🔬 Simulador**\n\nCambiar los numeros de un sprint hipotetico y ver como se mueve el riesgo.")
with col4:
    st.markdown("**🧠 Explicabilidad**\n\nQue variables pesan mas y por que las probabilidades son creibles.")
with col5:
    st.markdown("**📡 Salud**\n\nSi el modelo se esta degradando con el tiempo y si toca reentrenar.")

st.markdown(" ")
st.caption(
    f"Champion: {met.get('modelo_familia','')} · "
    f"{met.get('registry_name','')} {met.get('registry_version','')} · "
    f"Evaluado: {met.get('fecha_evaluacion','')}"
)
