"""Exploracion de las fuentes de datos del proyecto."""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils.eda_data import *
from utils.estilo import (aplicar_estilo, AZUL_CORP, GRIS_MEDIO, GRIS_FUERTE,
                          ROJO_ATENCION, AMBAR_MEDIO, VERDE_OK, GRIS_SUAVE)

st.set_page_config(page_title="Exploracion · ShiftMetrics", page_icon="📊", layout="wide")

st.markdown("## 📊 Exploracion de las fuentes de datos")
st.markdown(
    "Estos son los hallazgos principales de los 4 EDAs que hicimos antes de modelar. "
    "Cada fuente aporta un pedazo distinto de la historia."
)
st.divider()

fuente = st.radio(
    "Fuente de datos",
    ["PROMISE", "Apache JIRA", "Red Hat JIRA", "GHArchive", "Vista general"],
    horizontal=True,
)

if fuente == "PROMISE":
    st.markdown("### PROMISE — metricas de codigo fuente")
    st.markdown(
        "144 CSVs con metricas CK de proyectos open source. "
        "15,775 modulos de 41 proyectos. La cobertura contra Apache JIRA es del 1.6% — "
        "muy baja, pero las correlaciones con defectos son reales."
    )
    tab1, tab2, tab3 = st.tabs(["Balance de clases", "Correlaciones CK", "Densidad de defectos"])
    with tab1:
        col1, col2 = st.columns([2, 3])
        with col1:
            fig = go.Figure(go.Pie(
                labels=list(PROMISE_BALANCE.keys()),
                values=list(PROMISE_BALANCE.values()),
                marker_colors=[VERDE_OK, ROJO_ATENCION],
                hole=0.45, textinfo='label+percent'))
            fig.update_layout(title="Modulos con y sin defectos", showlegend=False)
            aplicar_estilo(fig, alto=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            total = sum(PROMISE_BALANCE.values())
            defectuosos = PROMISE_BALANCE["Defectuoso (1)"]
            st.markdown(
                f"De {total:,} modulos analizados, el **{defectuosos/total*100:.1f}%** "
                f"tiene al menos un bug reportado. Un modelo que diga 'todo limpio' "
                f"ya tiene 63.7% de accuracy. El reto es encontrar los defectuosos "
                f"sin generar demasiadas alertas falsas.")
    with tab2:
        corr = PROMISE_CK_CORR.copy()
        n_mostrar = st.slider("Metricas a mostrar", 5, 20, 10, key="ck_slider")
        corr_top = corr.head(n_mostrar)
        colores = [ROJO_ATENCION if v > 0 else AZUL_CORP for v in corr_top["correlacion"]]
        fig = go.Figure(go.Bar(
            x=corr_top["correlacion"][::-1], y=corr_top["metrica"][::-1],
            orientation="h", marker_color=colores[::-1],
            text=[f"{v:.3f}" for v in corr_top["correlacion"][::-1]], textposition="outside"))
        fig.update_layout(title="Correlacion Pearson: metricas CK vs presencia de defecto",
                          xaxis_title="Correlacion")
        aplicar_estilo(fig, alto=max(300, n_mostrar * 30))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "Las correlaciones son moderadas (la mas alta es RFC con 0.20). "
            "Ninguna metrica sola predice bien, pero combinadas con features de proceso aportan señal. "
            "**CAM tiene correlacion negativa** — mas cohesion de clase, menos bugs.")
    with tab3:
        dd = PROMISE_DEFECT_DENSITY.sort_values("densidad", ascending=False)
        top_n = st.slider("Proyectos a mostrar", 5, 41, 15, key="dd_slider")
        dd_top = dd.head(top_n)
        fig = go.Figure(go.Bar(
            x=dd_top["densidad"], y=dd_top["proyecto"], orientation="h",
            marker_color=[ROJO_ATENCION if d > 0.5 else AMBAR_MEDIO if d > 0.2 else VERDE_OK
                          for d in dd_top["densidad"]],
            text=[f"{d:.0%}" for d in dd_top["densidad"]], textposition="outside"))
        fig.update_layout(title="Densidad de defectos por proyecto",
                          xaxis_title="Proporcion de modulos con bugs",
                          yaxis=dict(autorange="reversed"))
        aplicar_estilo(fig, alto=max(300, top_n * 28))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"xalan-2.7 tiene el 98.8% de sus modulos con algun bug. "
            f"La mediana entre los {len(dd)} proyectos es {dd['densidad'].median():.0%}.")

elif fuente == "Apache JIRA":
    st.markdown("### Apache JIRA — fuente principal de sprints")
    st.markdown("6 colecciones de datos parquet. La tabla de issues es la base del target y features de conteo.")
    tab1, tab2, tab3 = st.tabs(["Colecciones", "Timeline", "Cycle Time"])
    with tab1:
        fig = go.Figure(go.Bar(x=APACHE_COLECCIONES["coleccion"], y=APACHE_COLECCIONES["filas"],
                               marker_color=AZUL_CORP,
                               text=[f"{f:,}" for f in APACHE_COLECCIONES["filas"]], textposition="outside"))
        fig.update_layout(title="Filas por coleccion", yaxis_title="Filas", yaxis_type="log")
        aplicar_estilo(fig, alto=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Issues tiene 112 columnas, pero solo 65 estan completas (>50% nulos en el resto).")
    with tab2:
        fig = go.Figure(go.Bar(x=APACHE_ISSUES_POR_ANO["ano"], y=APACHE_ISSUES_POR_ANO["issues"],
                               marker_color=AZUL_CORP, text=APACHE_ISSUES_POR_ANO["issues"], textposition="outside"))
        fig.update_layout(title="Issues creados por ano (muestra)", xaxis_title="Ano", yaxis_title="Issues", xaxis=dict(dtick=1))
        aplicar_estilo(fig, alto=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Los gaps (2007-08, 2012-13) son del muestreo del EDA. El dataset completo en Gold cubre 2000-2021.")
    with tab3:
        ct = APACHE_CYCLE_TIME
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Con resolucion", f"{ct['con_resolucion']:,} / {ct['total']:,}")
        c2.metric("Mediana", f"{ct['mediana']:.0f} dias")
        c3.metric("Media", f"{ct['media']:.0f} dias")
        c4.metric("Max", f"{ct['max']:,.0f} dias")
        st.markdown(f"Mediana {ct['mediana']:.0f} pero media {ct['media']:.0f} — cola pesada. Max {ct['max']:,.0f} dias (~9.6 anos). log1p obligatorio.")

elif fuente == "Red Hat JIRA":
    st.markdown("### Red Hat JIRA — tiempos de ciclo a escala")
    st.markdown(f"{REDHAT_META['filas']:,} issues de {REDHAT_META['proyectos']} proyectos ({REDHAT_META['rango_created'][0]} a {REDHAT_META['rango_created'][1]}).")
    tab1, tab2, tab3 = st.tabs(["Tipos de issue", "Estados", "Cycle Time"])
    with tab1:
        n_tipos = st.slider("Tipos a mostrar", 3, 10, 8, key="rh_tipos")
        top = REDHAT_ISSUE_TYPES.head(n_tipos)
        fig = go.Figure(go.Bar(y=top["tipo"][::-1], x=top["conteo"][::-1], orientation="h",
                               marker_color=[ROJO_ATENCION if t=="Bug" else AZUL_CORP for t in top["tipo"][::-1]],
                               text=[f"{c:,}" for c in top["conteo"][::-1]], textposition="outside"))
        fig.update_layout(title="Tipos de issue mas frecuentes", xaxis_title="Cantidad")
        aplicar_estilo(fig, alto=max(300, n_tipos*35))
        st.plotly_chart(fig, use_container_width=True)
        bugs_pct = REDHAT_ISSUE_TYPES.iloc[0]["conteo"] / REDHAT_META["filas"] * 100
        st.markdown(f"Bugs son el {bugs_pct:.1f}% de los issues. La proporcion Bug/Task da idea de cuanta carga es correctiva.")
    with tab2:
        n_est = st.slider("Estados a mostrar", 3, 15, 10, key="rh_est")
        top = REDHAT_STATUSES.head(n_est)
        fig = go.Figure(go.Bar(y=top["estado"][::-1], x=top["conteo"][::-1], orientation="h",
                               marker_color=AZUL_CORP, text=[f"{c:,}" for c in top["conteo"][::-1]], textposition="outside"))
        fig.update_layout(title="Estados de issues", xaxis_title="Cantidad")
        aplicar_estilo(fig, alto=max(300, n_est*30))
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        ct = REDHAT_CYCLE_TIME
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Cobertura", f"{ct['cobertura_pct']:.1f}%")
        c2.metric("p50", f"{ct['p50']:.0f} dias")
        c3.metric("p90", f"{ct['p90']:.0f} dias")
        c4.metric("Max", f"{ct['max']:,.0f} dias")
        fig = go.Figure()
        for label, val, color in [("p50",ct["p50"],VERDE_OK),("p75",ct["p75"],AMBAR_MEDIO),("p90",ct["p90"],ROJO_ATENCION)]:
            fig.add_trace(go.Bar(x=[label], y=[val], name=label, marker_color=color,
                                 text=[f"{val:.0f}d"], textposition="outside"))
        fig.update_layout(title="Percentiles del Cycle Time", yaxis_title="Dias", showlegend=False)
        aplicar_estilo(fig, alto=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"p50={ct['p50']:.0f}d pero p90={ct['p90']:.0f}d. Max {ct['max']:,.0f}d (~19 anos). log1p + flag cycle_missing.")

elif fuente == "GHArchive":
    st.markdown("### GHArchive — actividad GitHub 2022")
    st.markdown(f"{GHARCHIVE_META['archivos_json']} JSON.gz ({GHARCHIVE_META['tamano_gib']:.2f} GiB). {GHARCHIVE_META['eventos_muestra']:,} eventos Apache. Proxies DORA.")
    tab1, tab2, tab3 = st.tabs(["Tipos de evento", "Deploy Frequency", "Change Failure Rate"])
    with tab1:
        evt = GHARCHIVE_EVENT_TYPES.copy()
        evt["pct"] = evt["conteo"] / evt["conteo"].sum() * 100
        dora_types = {"PushEvent","PullRequestEvent","CreateEvent","ReleaseEvent"}
        colores = [AZUL_CORP if t in dora_types else GRIS_MEDIO for t in evt["tipo"]]
        fig = go.Figure(go.Bar(y=evt["tipo"][::-1], x=evt["conteo"][::-1], orientation="h",
                               marker_color=colores[::-1],
                               text=[f"{c:,} ({p:.1f}%)" for c,p in zip(evt["conteo"][::-1], evt["pct"][::-1])],
                               textposition="outside"))
        fig.update_layout(title="Tipos de evento GHArchive 2022", xaxis_title="Cantidad")
        aplicar_estilo(fig, alto=450)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("En azul los relevantes para DORA (PushEvent→deploy freq, PullRequestEvent→CFR).")
    with tab2:
        n_repos = st.slider("Repos a mostrar", 5, 13, 10, key="gh_repos")
        top = GHARCHIVE_DEPLOY_FREQ.head(n_repos)
        fig = go.Figure(go.Bar(y=top["repo"][::-1], x=top["pushes_main"][::-1], orientation="h",
                               marker_color=AZUL_CORP, text=top["pushes_main"][::-1], textposition="outside"))
        fig.update_layout(title="Pushes a main/master por repo", xaxis_title="Pushes")
        aplicar_estilo(fig, alto=max(300, n_repos*30))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"apache/camel lidera con {top.iloc[0]['pushes_main']} pushes. Total repos con pushes: {GHARCHIVE_META['repos_apache']}.")
    with tab3:
        cfr = GHARCHIVE_CFR
        c1,c2,c3 = st.columns(3)
        c1.metric("PRs merged", f"{cfr['prs_merged']}")
        c2.metric("Cerrados sin merge", f"{cfr['prs_cerrados_sin_merge']}")
        c3.metric("CFR proxy", f"{cfr['cfr_proxy']:.1%}")
        fig = go.Figure(go.Pie(
            labels=["Merged","Cerrado sin merge","Abiertos/otros"],
            values=[cfr["prs_merged"], cfr["prs_cerrados_sin_merge"],
                    cfr["total_prs"]-cfr["prs_merged"]-cfr["prs_cerrados_sin_merge"]],
            marker_colors=[VERDE_OK, ROJO_ATENCION, GRIS_SUAVE],
            hole=0.45, textinfo='label+percent'))
        fig.update_layout(title="Descomposicion de Pull Requests")
        aplicar_estilo(fig, alto=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"CFR proxy = {cfr['cfr_proxy']:.1%}. No es CFR real (haria falta medir rollbacks en produccion).")
    st.markdown("#### Distribucion horaria")
    fig = go.Figure(go.Bar(x=GHARCHIVE_TEMPORAL["hora"], y=GHARCHIVE_TEMPORAL["eventos"],
                           marker_color=AZUL_CORP, text=GHARCHIVE_TEMPORAL["eventos"], textposition="outside"))
    fig.update_layout(title="Eventos Apache por hora (UTC)", xaxis_title="Hora UTC", yaxis_title="Eventos",
                      xaxis=dict(tickvals=[0,6,12,18], ticktext=["00h","06h","12h","18h"]))
    aplicar_estilo(fig, alto=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Pico a las 12h UTC — horario laboral europeo y costa este de EE.UU.")

elif fuente == "Vista general":
    st.markdown("### Panorama de las 4 fuentes")
    resumen = pd.DataFrame({
        "Fuente": ["PROMISE","Apache JIRA","Red Hat JIRA","GHArchive"],
        "Filas": ["15,775","978K (full)","505,096","4,994"],
        "Periodo": ["N/A","2003-2020","2001-2024","Ene-Mar 2022"],
        "Aporta": ["Metricas CK (1.6%)","Target + bugs + stories","Cycle time","Deploy freq + CFR (9%)"],
    })
    st.dataframe(resumen, use_container_width=True, hide_index=True)
    fig = go.Figure(go.Bar(
        x=["PROMISE","Apache JIRA\n(muestra)","Red Hat JIRA","GHArchive\n(Apache)"],
        y=[15775, 1875, 505096, 4994],
        marker_color=[GRIS_MEDIO, AZUL_CORP, AZUL_CORP, GRIS_MEDIO],
        text=["15.8K","1.9K","505K","5.0K"], textposition="outside"))
    fig.update_layout(title="Tamano de cada fuente (filas EDA)", yaxis_title="Filas", yaxis_type="log")
    aplicar_estilo(fig, alto=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""**Lo que nos llevamos al modelo:**
- 70% positivos → baseline trivial da F2=0.916. Hay que ganar con precision.
- Drift temporal: cycle time bajo 96% entre 2008 y 2021.
- Cobertura cruzada baja: PROMISE 1.6%, DORA 9%. Los flags de ausencia capturan esa senal.
- log1p obligatorio: cycle time tiene p50=28d y max=7,030d.""")
