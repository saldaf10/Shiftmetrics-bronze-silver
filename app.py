"""ShiftMetrics — inicio."""
from utils.theme import aplicar_tema, C, hero, badge, nota, separador, stat_card, seccion_titulo
aplicar_tema("ShiftMetrics", "⚡")

import streamlit as st
from utils.data import cargar_metricas, get_data_source

met = cargar_metricas()
src = get_data_source()

# Sidebar
with st.sidebar:
    st.markdown(
        f"<div style='text-align:center; padding:16px 0;'>"
        f"<div style='font-size:32px;'>⚡</div>"
        f"<div style='font-size:20px; font-weight:900; "
        f"background:{C['grad_brand']}; -webkit-background-clip:text; "
        f"-webkit-text-fill-color:transparent;'>ShiftMetrics</div>"
        f"<div style='font-size:11px; color:{C['text3']}; margin-top:2px; "
        f"letter-spacing:0.1em; text-transform:uppercase;'>sprint defect prediction</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    if src["connected"]:
        st.success("📡 MLflow live")
    else:
        st.info("📁 Offline mode")
    st.markdown("---")
    st.caption(f"Registry: {met.get('registry_name','')}")
    st.caption(f"Version: {met.get('registry_version','')}")
    st.caption(f"Source: {met.get('source','local')}")

# Hero
badges_html = (
    badge(f"F2 {met.get('f2',0):.3f}", "success") + " "
    + badge(f"RECALL {met.get('recall',0):.3f}", "success") + " "
    + badge(f"BRIER {met.get('brier',0):.4f}", "info") + " "
    + badge(met.get("source","LOCAL").upper(), "brand")
)
hero("ShiftMetrics",
     "Identificar sprints con alto riesgo de dejar escapar defectos a produccion. "
     "Pipeline end-to-end desde datos crudos de Apache JIRA y GitHub hasta prediccion calibrada.",
     badges_html)

# KPIs grandes
c1, c2, c3, c4 = st.columns(4)
with c1: stat_card(f"{met.get('f2',0):.3f}", "F2-score", "metrica principal", "brand")
with c2: stat_card(f"{met.get('recall',0):.3f}", "Recall", "cobertura de bugs", "green")
with c3: stat_card(f"{met.get('precision',0):.3f}", "Precision", "calidad de alertas", "cyan")
with c4: stat_card(f"{met.get('brier',0):.4f}", "Brier", "calibracion", "amber")

separador()

# Pregunta de oro
st.markdown(
    f"<div style='background:{C['bg2']}; border:1px solid {C['brand']}40; "
    f"border-radius:14px; padding:28px 32px; text-align:center; "
    f"box-shadow: 0 0 30px {C['brand']}15;'>"
    f"<div style='color:{C['brand_lt']}; font-size:11px; font-weight:700; "
    f"text-transform:uppercase; letter-spacing:0.15em; margin-bottom:10px;'>Pregunta del proyecto</div>"
    f"<div style='font-size:20px; color:{C['text_bright']}; font-weight:700; line-height:1.5;'>"
    f"¿En cuales sprints hay que reforzar QA antes del cierre "
    f"para que no se escape un defecto a produccion?</div>"
    f"</div>",
    unsafe_allow_html=True
)

separador()

# Secciones
seccion_titulo("Secciones del tablero", "Cada pagina responde una pregunta distinta")

secciones = [
    ("📊", "Exploracion", "EDAs de las 4 fuentes del pipeline Bronze", "brand"),
    ("🚦", "Riesgo", "Ranking de sprints por probabilidad de defecto", "red"),
    ("🔬", "Simulador", "Probar un sprint hipotetico y ver que lo mueve", "cyan"),
    ("🧠", "Explicabilidad", "Variables clave y calibracion del modelo", "brand"),
    ("📡", "Salud", "Drift, degradacion y cuando reentrenar", "amber"),
]

cols = st.columns(len(secciones))
for col, (icon, title, desc, color) in zip(cols, secciones):
    accent = {"brand":C["brand"],"red":C["red"],"cyan":C["cyan"],"amber":C["amber"]}.get(color,C["brand"])
    with col:
        st.markdown(
            f"<div style='background:{C['bg2']}; border:1px solid {C['border']}; "
            f"border-radius:12px; padding:20px; min-height:140px; "
            f"border-top:3px solid {accent};'>"
            f"<div style='font-size:28px; margin-bottom:8px;'>{icon}</div>"
            f"<div style='font-weight:800; color:{C['text_bright']}; font-size:14px; "
            f"margin-bottom:4px;'>{title}</div>"
            f"<div style='color:{C['text3']}; font-size:12px; line-height:1.5;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

separador()
nota("Las paginas van en orden narrativo desde el sidebar. Cada una funciona independiente.", "💡")

separador()
st.markdown(
    f"<div style='text-align:center; color:{C['text3']}; font-size:11px; "
    f"font-family:JetBrains Mono,monospace; padding:12px;'>"
    f"{met.get('modelo_familia','')} · {met.get('fecha_evaluacion','')}</div>",
    unsafe_allow_html=True
)
