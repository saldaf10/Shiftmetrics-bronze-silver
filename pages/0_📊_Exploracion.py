"""Exploracion inicial — graficos de los 4 EDAs del proyecto."""
from __future__ import annotations
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Exploracion · ShiftMetrics", page_icon="📊", layout="wide")

st.markdown("## 📊 Exploracion de los datos")
st.markdown(
    "Antes de construir el modelo hicimos un EDA por cada fuente. "
    "Aca quedan los graficos principales que nos llevaron a las decisiones de features."
)
st.divider()

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "notebooks" / "outputs"

def img(nombre, caption=""):
    p = IMG / nombre
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"No se encontro: {nombre}")


# ─────────────────────────────────────────────
st.markdown("### 1. PROMISE — metricas de codigo fuente")
st.markdown(
    "144 CSVs con metricas CK de proyectos open source. "
    "La cobertura contra Apache JIRA es bajisima (1.6%), "
    "pero el EDA muestra que las metricas de complejidad si correlacionan con defectos."
)

col1, col2 = st.columns(2)
with col1:
    img("promise_balance_clases.png", "Balance de clases (con/sin bugs) en los proyectos PROMISE")
with col2:
    img("promise_defect_density.png", "Densidad de defectos por proyecto")

col3, col4 = st.columns(2)
with col3:
    img("promise_metricas_ck.png", "Distribucion de metricas CK (WMC, CBO, RFC, LCOM)")
with col4:
    img("promise_correlaciones.png", "Correlacion entre metricas CK y conteo de defectos")

img("promise_metricas_mccabe.png", "Metricas de complejidad McCabe")

st.markdown(" ")


# ─────────────────────────────────────────────
st.markdown("### 2. Apache JIRA — fuente principal")
st.markdown(
    "978K issues de 42 proyectos Apache. Esta tabla define el target y las features "
    "de conteo (bugs, stories, tasks). El hallazgo clave: el 70% de los sprints "
    "tiene al menos un defecto escapado — eso es el baseline que hay que superar."
)

col1, col2 = st.columns(2)
with col1:
    img("apache_completitud_campos.png", "Completitud de campos por coleccion")
with col2:
    img("apache_distribucion_temporal.png", "Distribucion temporal de issues")

col3, col4 = st.columns(2)
with col3:
    img("apache_cycle_time_issues.png", "Cycle time calculado directo desde issues")
with col4:
    img("apache_cycle_time_events.png", "Cycle time desde transiciones de estado (events)")

st.markdown(" ")


# ─────────────────────────────────────────────
st.markdown("### 3. Red Hat JIRA — tiempos de ciclo")
st.markdown(
    "Datos de ciclo de vida de Red Hat. Lo importante: la distribucion de cycle time "
    "es log-normal con cola pesada (p50=90 dias, p99=2341 dias). "
    "Eso justifica aplicar log1p antes de meterlo al modelo."
)

col1, col2 = st.columns(2)
with col1:
    img("redhat_cycle_time_dist.png", "Distribucion del cycle time")
with col2:
    img("redhat_resolucion_temporal.png", "Resolucion de issues por periodo")

img("redhat_prioridad.png", "Distribucion por prioridad de issue")

st.markdown(" ")


# ─────────────────────────────────────────────
st.markdown("### 4. GHArchive — metricas DORA")
st.markdown(
    "Eventos de GitHub del 2022 de repos Apache. Sacamos deploy frequency "
    "y change failure rate. Solo el 9% de los sprints tiene esta info "
    "por el mismatch temporal con JIRA (sprints hasta 2021, GitHub 2022)."
)

col1, col2 = st.columns(2)
with col1:
    img("gharchive_event_types.png", "Tipos de eventos en GHArchive")
with col2:
    img("gharchive_deployment_freq.png", "Frecuencia de deploy por repo")

img("gharchive_temporal.png", "Distribucion temporal de eventos 2022")


st.markdown(" ")
st.divider()


# ─────────────────────────────────────────────
st.markdown("### Lo que nos llevamos al modelo")
st.markdown("""
- **70% positivos** — el baseline trivial ya da F2 de 0.916. Cualquier modelo tiene que ganar con precision, no con recall.
- **Drift real**: el cycle time bajo 96% entre 2008 y 2021. El modelo necesita `sprint_year` para adaptarse.
- **Cobertura cruzada baja**: PROMISE 1.6%, DORA 9%. No es error del pipeline, es limitacion del dominio. Los flags de ausencia capturan esa senal.
- **log1p obligatorio**: cycle time y bug/story ratio tienen colas tan pesadas que sin transformar dominan el gradiente.
""")
