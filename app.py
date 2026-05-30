"""ShiftMetrics — inicio."""
from utils.theme import aplicar_tema, C, hero, badge, nota, separador
aplicar_tema("ShiftMetrics", "⚡")

import streamlit as st
from utils.data import cargar_metricas, get_data_source

met = cargar_metricas()
src = get_data_source()

# Sidebar
with st.sidebar:
    st.markdown("# ⚡ ShiftMetrics")
    st.caption("sprint defect prediction")
    st.markdown("---")
    if src["connected"]:
        st.success("📡 MLflow live")
        st.caption(src.get("url", ""))
    else:
        st.info("📁 Offline mode")
    st.markdown("---")
    st.markdown(
        f"**Registry**  \n`{met.get('registry_name','')}`  \n"
        f"**Version** `{met.get('registry_version','')}`  \n"
        f"**Fuente** `{met.get('source','local')}`"
    )

# Hero con identidad
badges_html = (
    badge(f"F2 {met.get('f2',0):.3f}", "success") + " "
    + badge(f"Recall {met.get('recall',0):.3f}", "success") + " "
    + badge(f"Brier {met.get('brier',0):.4f}", "info") + " "
    + badge(met.get("source","local").upper(), "brand")
)
hero("ShiftMetrics",
     "Predecir en cuales sprints Apache van a escaparse bugs a produccion",
     badges_html)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("F2-score", f"{met.get('f2',0):.3f}",
          help="Metrica principal del proyecto. Penaliza mas los falsos negativos.")
c2.metric("Recall", f"{met.get('recall',0):.3f}",
          help="Cuantos sprints con bug real se lograron marcar.")
c3.metric("Precision", f"{met.get('precision',0):.3f}",
          help="De las alertas emitidas, cuantas acertaron.")
c4.metric("Brier score", f"{met.get('brier',0):.4f}",
          help="Calibracion de probabilidades. Mas cerca de 0 = mas confiable.")

separador()

# Pregunta
st.markdown(
    f"<div style='background:{C['bg_light']}; border:2px solid {C['brand']}; "
    f"border-radius:12px; padding:24px 28px; text-align:center;'>"
    f"<div style='color:{C['text3']}; font-size:12px; text-transform:uppercase; "
    f"font-weight:700; letter-spacing:0.08em; margin-bottom:8px;'>Pregunta del proyecto</div>"
    f"<div style='font-size:18px; color:{C['text']}; font-weight:600; line-height:1.5;'>"
    f"¿En cuales sprints conviene reforzar la revision de calidad "
    f"antes del cierre para que no escape un defecto a produccion?</div>"
    f"</div>",
    unsafe_allow_html=True
)

separador()

# Navegacion
st.markdown("### Secciones del tablero")
secciones = [
    ("📊", "Exploracion", "Los 4 EDAs del pipeline Bronze — PROMISE, Apache JIRA, Red Hat y GHArchive."),
    ("🚦", "Riesgo", "Ranking de sprints ordenados por probabilidad. La decision de QA empieza aca."),
    ("🔬", "Simulador", "Sprint hipotetico — mover numeros y ver como cambia el riesgo."),
    ("🧠", "Explicabilidad", "Que variables pesan, por que las probabilidades son creibles."),
    ("📡", "Salud", "Drift, degradacion y cuando reentrenar."),
]

cols = st.columns(len(secciones))
for col, (icon, title, desc) in zip(cols, secciones):
    with col:
        st.markdown(
            f"<div style='background:{C['bg_light']}; border:1px solid {C['border_lt']}; "
            f"border-radius:10px; padding:18px; min-height:150px;'>"
            f"<div style='font-size:26px; margin-bottom:6px;'>{icon}</div>"
            f"<div style='font-weight:700; color:{C['text']}; font-size:14px; margin-bottom:4px;'>{title}</div>"
            f"<div style='color:{C['text2']}; font-size:13px; line-height:1.5;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

separador()
nota("Las paginas estan en orden narrativo en el sidebar. Cada una se puede leer independiente.")

separador()
st.markdown(
    f"<div style='text-align:center; color:{C['text3']}; font-size:12px;'>"
    f"ShiftMetrics · {met.get('modelo_familia','')} · "
    f"Evaluado: {met.get('fecha_evaluacion','')}</div>",
    unsafe_allow_html=True
)
