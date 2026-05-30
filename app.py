"""ShiftMetrics — pagina de inicio."""
from utils.theme import aplicar_tema, COLORS, card, badge, stat_grande, nota_lateral, separador
aplicar_tema("ShiftMetrics", "⚡")

import streamlit as st
from utils.data import cargar_metricas, get_data_source

met = cargar_metricas()
src = get_data_source()

# Sidebar branding
with st.sidebar:
    st.markdown("# ⚡ ShiftMetrics")
    st.markdown("Prediccion de defectos escapados en sprints Apache.")
    st.markdown("---")
    if src["connected"]:
        st.success("📡 MLflow conectado")
    else:
        st.info("📁 Modo offline")
    st.markdown("---")
    st.caption(f"Registry: {met.get('registry_name','')} {met.get('registry_version','')}")

# Hero
st.markdown(
    f"<div style='background: linear-gradient(135deg, {COLORS['brand']} 0%, #312e81 100%); "
    f"border-radius:20px; padding:40px 48px; margin-bottom:24px;'>"
    f"<h1 style='color:white; margin:0; font-size:42px;'>⚡ ShiftMetrics</h1>"
    f"<p style='color:#c7d2fe; font-size:18px; margin-top:8px; max-width:700px;'>"
    f"Saber en cuales sprints concentrar QA antes del cierre "
    f"para que los bugs no se escapen a produccion.</p>"
    f"<div style='margin-top:16px;'>"
    + badge(f"F2 = {met.get('f2',0):.3f}", "success") + "  "
    + badge(f"Recall = {met.get('recall',0):.3f}", "success") + "  "
    + badge(f"Fuente: {met.get('source','local')}", "brand")
    + f"</div></div>",
    unsafe_allow_html=True
)

separador()

# KPIs principales — con espacio
c1, c2, c3, c4 = st.columns(4)
c1.metric("F2-score", f"{met.get('f2',0):.3f}",
          help="Metrica principal. Penaliza mas dejar pasar un bug que dar alerta falsa.")
c2.metric("Recall", f"{met.get('recall',0):.3f}",
          help="De los sprints con bug real, cuantos logro marcar.")
c3.metric("Precision", f"{met.get('precision',0):.3f}",
          help="De las alertas emitidas, cuantas acertaron.")
c4.metric("Brier score", f"{met.get('brier',0):.3f}",
          help="Calibracion de probabilidades. Menor es mejor.")

separador()

# Pregunta de oro
st.markdown("### 🎯 La pregunta que resolvemos")
st.markdown(
    f"<div style='background:{COLORS['brand_lt']}; border:2px solid {COLORS['brand']}; "
    f"border-radius:14px; padding:24px 28px; font-size:18px; color:{COLORS['text']}; "
    f"line-height:1.5; text-align:center;'>"
    f"<b>¿En cuales sprints hay que reforzar la revision de calidad "
    f"antes del cierre para evitar que un defecto llegue a produccion?</b>"
    f"</div>",
    unsafe_allow_html=True
)

separador()

# Navegacion con cards
st.markdown("### Recorrido del tablero")

col1, col2, col3, col4, col5 = st.columns(5)

sections = [
    (col1, "📊", "Exploracion", "Graficos de los 4 EDAs iniciales — PROMISE, Apache JIRA, Red Hat y GHArchive."),
    (col2, "🚦", "Riesgo", "Ranking de sprints por probabilidad. La decision empieza aca."),
    (col3, "🔬", "Simulador", "Probar un sprint hipotetico y ver que mueve el riesgo."),
    (col4, "🧠", "Explicabilidad", "Que variables pesan y por que las probabilidades son creibles."),
    (col5, "📡", "Salud", "Si el modelo se esta degradando y si toca reentrenar."),
]

for col, icon, title, desc in sections:
    with col:
        st.markdown(
            f"<div style='background:{COLORS['bg3']}; border:1px solid {COLORS['border']}; "
            f"border-radius:14px; padding:20px; min-height:160px;'>"
            f"<div style='font-size:28px; margin-bottom:8px;'>{icon}</div>"
            f"<div style='font-weight:700; color:{COLORS['text']}; font-size:15px; margin-bottom:6px;'>{title}</div>"
            f"<div style='color:{COLORS['text2']}; font-size:13px; line-height:1.5;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

separador()

nota_lateral(
    "Las paginas estan en orden narrativo en el menu lateral, "
    "pero cada una se puede leer independiente."
)

separador()

# Footer
st.markdown(
    f"<div style='text-align:center; color:{COLORS['text3']}; font-size:12px; padding:16px 0;'>"
    f"ShiftMetrics · {met.get('modelo_familia','')} · "
    f"Evaluado: {met.get('fecha_evaluacion','')} · "
    f"Fuente: {'MLflow' if met.get('source')=='mlflow' else 'datos locales'}"
    f"</div>",
    unsafe_allow_html=True
)
