"""ShiftMetrics Dashboard — Dash version.
Sprint defect prediction for Apache open-source ecosystem.
Dark theme, GitHub-inspired, interactive, animated.
"""
from __future__ import annotations
import json, math, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, dash_table, callback, no_update

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
DATA = Path(__file__).parent / "data"

def load_json(name):
    with open(DATA / name) as f: return json.load(f)

def load_parquet(name):
    return pd.read_parquet(DATA / name)

def load_model():
    with open(DATA / "champion.pkl", "rb") as f: return pickle.load(f)

MET = load_json("model_metrics.json")
SNAP = load_parquet("test_snapshot.parquet")
SHAP_G = load_parquet("shap_global.parquet")
DRIFT_PSI = load_parquet("drift_psi.parquet")
DRIFT_Y = load_parquet("drift_yearly.parquet")
RELIABILITY = load_parquet("reliability_curve.parquet")
CHAMPION = load_model()

from eda_data import *

# Feature descriptions
FEAT_DESC = {
    "num_bugs_sprint":"Bugs","num_stories_sprint":"Stories","num_tasks_sprint":"Tasks",
    "total_issues_sprint":"Total issues","log_avg_cycle_time":"Cycle time (log)",
    "log_bug_story_ratio":"Bug/story ratio (log)","log_total_issues":"Sprint size (log)",
    "sprint_year":"Year","sprint_month_sin":"Month (sin)","sprint_month_cos":"Month (cos)",
    "deploy_frequency_weekly":"Deploy freq","change_failure_rate":"Change failure rate",
    "bsr_missing":"No bug/story ratio","cycle_missing":"No cycle time",
    "dora_missing":"No DORA metrics","bugs_per_issue":"Bug density",
    "log_cycle_x_bsr":"Chronic bugs",
}

FEATURE_COLS = CHAMPION.get("feature_cols", list(FEAT_DESC.keys()))
THRESHOLD = CHAMPION.get("threshold", 0.220)

# ═══════════════════════════════════════════════════════════════
# PLOTLY DARK STYLE
# ═══════════════════════════════════════════════════════════════
NEON_PURPLE = "#a78bfa"
NEON_CYAN   = "#22d3ee"
NEON_GREEN  = "#34d399"
NEON_RED    = "#f87171"
NEON_AMBER  = "#fbbf24"
NEON_BLUE   = "#60a5fa"
BG_DARK     = "#0d1117"
BG_CARD     = "#161b22"
BG_ELEVATED = "#1c2333"
BORDER      = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"

def dark_fig(fig, h=420):
    fig.update_layout(
        template="plotly_dark", height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.6)",
        margin=dict(l=50, r=30, t=50, b=50),
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=13),
        title_font=dict(color="#ffffff", size=17, family="Inter"),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=BORDER, borderwidth=1,
                    font=dict(size=12, color=TEXT_MUTED)),
    )
    fig.update_xaxes(gridcolor="#21262d", linecolor=BORDER, zeroline=False,
                     title_font=dict(color=TEXT_MUTED, size=12))
    fig.update_yaxes(gridcolor="#21262d", linecolor=BORDER, zeroline=False,
                     title_font=dict(color=TEXT_MUTED, size=12))
    return fig


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── RESET ── */
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', -apple-system, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
}}

/* ── OCTOCAT WATERMARK ── */
body::before {{
    content: '';
    position: fixed;
    bottom: -80px; right: -80px;
    width: 500px; height: 500px;
    background-image: url('https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png');
    background-size: contain; background-repeat: no-repeat;
    opacity: 0.03;
    pointer-events: none;
    z-index: 0;
}}

/* ── ANIMATED GRADIENT BG ── */
body::after {{
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 0%, rgba(137,87,229,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(34,211,238,0.06) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}}

.page-shell {{
    position: relative; z-index: 1;
    max-width: 1500px; margin: 0 auto;
    padding: 24px 32px 60px;
}}

/* ── HERO ── */
.hero {{
    background: linear-gradient(135deg, rgba(137,87,229,0.12) 0%, rgba(34,211,238,0.08) 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, {NEON_PURPLE}, {NEON_CYAN}, {NEON_GREEN});
    animation: gradientSlide 4s ease infinite;
}}
@keyframes gradientSlide {{
    0%,100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}
.hero-title {{
    font-size: 42px; font-weight: 900; line-height: 1;
    background: linear-gradient(135deg, {NEON_PURPLE}, {NEON_CYAN});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 8px;
}}
.hero-sub {{ color: {TEXT_MUTED}; font-size: 16px; max-width: 700px; line-height: 1.6; }}
.hero-badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }}

/* ── BADGES ── */
.badge {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.badge-purple {{ background: rgba(167,139,250,0.15); color: {NEON_PURPLE}; border: 1px solid rgba(167,139,250,0.3); }}
.badge-green  {{ background: rgba(52,211,153,0.15); color: {NEON_GREEN}; border: 1px solid rgba(52,211,153,0.3); }}
.badge-cyan   {{ background: rgba(34,211,238,0.15); color: {NEON_CYAN}; border: 1px solid rgba(34,211,238,0.3); }}
.badge-red    {{ background: rgba(248,113,113,0.15); color: {NEON_RED}; border: 1px solid rgba(248,113,113,0.3); }}
.badge-amber  {{ background: rgba(251,191,36,0.15); color: {NEON_AMBER}; border: 1px solid rgba(251,191,36,0.3); }}

/* ── METRIC CARDS ── */
.metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
.metric-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 24px; text-align: center;
    transition: all 0.3s ease;
}}
.metric-card:hover {{
    border-color: {NEON_PURPLE};
    box-shadow: 0 0 25px rgba(137,87,229,0.15);
    transform: translateY(-2px);
}}
.metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 36px; font-weight: 900; line-height: 1;
}}
.metric-label {{
    color: {TEXT_MUTED}; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 8px;
}}
.metric-detail {{ color: #4a5568; font-size: 11px; margin-top: 4px; }}

/* ── CARDS ── */
.card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 24px; margin: 16px 0;
}}
.card-glow {{
    box-shadow: 0 0 20px rgba(137,87,229,0.08);
    border-color: rgba(137,87,229,0.3);
}}
.card-title {{
    font-size: 18px; font-weight: 800; color: #fff; margin-bottom: 8px;
}}
.card-text {{ color: {TEXT_MUTED}; font-size: 14px; line-height: 1.6; }}

/* ── INSIGHT BOX ── */
.insight {{
    background: {BG_ELEVATED};
    border-left: 3px solid {NEON_PURPLE};
    border: 1px solid {BORDER};
    border-left: 3px solid {NEON_PURPLE};
    border-radius: 0 10px 10px 0;
    padding: 16px 20px; margin: 12px 0;
    color: {TEXT_MUTED}; font-size: 14px; line-height: 1.6;
}}

/* ── SECTION ── */
.section-header {{
    display: flex; align-items: center; gap: 12px;
    margin: 28px 0 16px;
}}
.section-icon {{ font-size: 28px; }}
.section-title {{ font-size: 22px; font-weight: 800; color: #fff; }}
.section-sub {{ color: {TEXT_MUTED}; font-size: 14px; margin-top: 2px; }}

/* ── GRID ── */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
.grid-5 {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }}

/* ── TABS ── */
.custom-tabs .tab {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px 10px 0 0 !important;
    color: {TEXT_MUTED} !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 12px 24px !important;
}}
.custom-tabs .tab--selected {{
    background: {BG_ELEVATED} !important;
    border-bottom: 3px solid {NEON_PURPLE} !important;
    color: #fff !important;
}}

/* ── TABLES ── */
.dash-table-container .dash-spreadsheet-container {{
    border-radius: 10px; overflow: hidden;
}}
.dash-table-container th {{
    background: {BG_ELEVATED} !important;
    color: {TEXT_PRIMARY} !important;
    font-weight: 700 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.05em;
    border-bottom: 2px solid {NEON_PURPLE} !important;
    padding: 12px 16px !important;
}}
.dash-table-container td {{
    background: {BG_CARD} !important;
    color: {TEXT_PRIMARY} !important;
    border-bottom: 1px solid {BORDER} !important;
    padding: 10px 16px !important; font-size: 13px !important;
}}
.dash-table-container tr:hover td {{
    background: {BG_ELEVATED} !important;
}}

/* ── DROPDOWNS / INPUTS ── */
.Select-control, .Select-menu-outer {{
    background: {BG_ELEVATED} !important;
    border-color: {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
}}
.Select-value-label {{ color: {TEXT_PRIMARY} !important; }}
input[type="number"] {{
    background: {BG_ELEVATED} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 8px; padding: 8px 12px;
}}

/* ── FOOTER ── */
.footer {{
    text-align: center; padding: 24px;
    color: #4a5568; font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid {BORDER}; margin-top: 40px;
}}

/* ── RESPONSIVE ── */
@media (max-width: 1100px) {{
    .metrics-grid, .grid-2, .grid-3, .grid-5 {{
        grid-template-columns: 1fr;
    }}
}}

/* ── PULSE ANIMATION ── */
@keyframes pulse {{
    0%,100% {{ opacity: 1; }}
    50% {{ opacity: 0.6; }}
}}
.pulse {{ animation: pulse 2s ease-in-out infinite; }}

/* ── GLOW ── */
.glow-purple {{ text-shadow: 0 0 10px rgba(167,139,250,0.5); }}
.glow-green  {{ text-shadow: 0 0 10px rgba(52,211,153,0.5); }}
.glow-cyan   {{ text-shadow: 0 0 10px rgba(34,211,238,0.5); }}
"""

# ═══════════════════════════════════════════════════════════════
# HELPER COMPONENTS
# ═══════════════════════════════════════════════════════════════
def metric_card(value, label, detail="", color=NEON_PURPLE):
    return html.Div([
        html.Div(value, className="metric-value", style={"color": color}),
        html.Div(label, className="metric-label"),
        html.Div(detail, className="metric-detail") if detail else None,
    ], className="metric-card")

def section(icon, title, sub=""):
    return html.Div([
        html.Span(icon, className="section-icon"),
        html.Div([
            html.Div(title, className="section-title"),
            html.Div(sub, className="section-sub") if sub else None,
        ]),
    ], className="section-header")

def insight(text):
    return html.Div(text, className="insight")

def badge(text, variant="purple"):
    return html.Span(text, className=f"badge badge-{variant}")

# ═══════════════════════════════════════════════════════════════
# TAB: HOME
# ═══════════════════════════════════════════════════════════════
def make_home():
    return html.Div([
        # Hero
        html.Div([
            html.Div("ShiftMetrics", className="hero-title"),
            html.Div(
                "Sprint defect prediction para el ecosistema Apache. "
                "Pipeline end-to-end desde datos crudos de JIRA y GitHub "
                "hasta prediccion calibrada con XGBoost.",
                className="hero-sub"
            ),
            html.Div([
                badge(f"F2 {MET['f2']:.3f}", "green"),
                badge(f"RECALL {MET['recall']:.3f}", "green"),
                badge(f"BRIER {MET['brier']:.4f}", "cyan"),
                badge(f"THRESHOLD {THRESHOLD:.3f}", "purple"),
                badge(f"6,683 sprints evaluados", "amber"),
            ], className="hero-badges"),
        ], className="hero"),

        # KPIs
        html.Div([
            metric_card(f"{MET['f2']:.3f}", "F2-SCORE", "metrica principal del proyecto", NEON_PURPLE),
            metric_card(f"{MET['recall']:.3f}", "RECALL", "cobertura de defectos reales", NEON_GREEN),
            metric_card(f"{MET['precision']:.3f}", "PRECISION", "calidad de las alertas", NEON_CYAN),
            metric_card(f"{MET['brier']:.4f}", "BRIER SCORE", "calibracion de probabilidades", NEON_AMBER),
        ], className="metrics-grid"),

        # Pregunta de oro
        html.Div([
            html.Div("PREGUNTA DEL PROYECTO", style={
                "color": NEON_PURPLE, "fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "0.15em", "marginBottom": "10px"
            }),
            html.Div(
                "¿En cuales sprints conviene reforzar la revision de calidad "
                "antes del cierre para evitar que un defecto escape a produccion?",
                style={"fontSize": "20px", "fontWeight": "700", "color": "#fff", "lineHeight": "1.5"}
            ),
        ], className="card card-glow", style={"textAlign": "center"}),

        # Pipeline status
        section("🔧", "Pipeline de datos", "De fuentes crudas a prediccion calibrada"),
        html.Div([
            html.Div([
                html.Div("BRONZE", style={"color": NEON_AMBER, "fontWeight": "800", "fontSize": "13px"}),
                html.Div("4 fuentes", style={"fontSize": "24px", "fontWeight": "900", "color": "#fff"}),
                html.Div("PROMISE + Apache JIRA + Red Hat + GHArchive", className="card-text"),
            ], className="metric-card"),
            html.Div([
                html.Div("SILVER", style={"color": NEON_CYAN, "fontWeight": "800", "fontSize": "13px"}),
                html.Div("978K", style={"fontSize": "24px", "fontWeight": "900", "color": "#fff"}),
                html.Div("issues procesados y validados", className="card-text"),
            ], className="metric-card"),
            html.Div([
                html.Div("GOLD", style={"color": NEON_GREEN, "fontWeight": "800", "fontSize": "13px"}),
                html.Div("42,747", style={"fontSize": "24px", "fontWeight": "900", "color": "#fff"}),
                html.Div("sprints con 17 features engineered", className="card-text"),
            ], className="metric-card"),
            html.Div([
                html.Div("ML", style={"color": NEON_PURPLE, "fontWeight": "800", "fontSize": "13px"}),
                html.Div("XGBoost", style={"fontSize": "24px", "fontWeight": "900", "color": "#fff"}),
                html.Div("calibrado + SHAP + drift monitoring", className="card-text"),
            ], className="metric-card"),
        ], className="metrics-grid"),

        insight(
            "El modelo no reemplaza al equipo de QA — lo que hace es ordenar la cola de trabajo. "
            "En vez de revisar 6,683 sprints, el equipo puede focalizarse en los ~300 que el modelo "
            "marca como alto riesgo y cubrir el 95% de los defectos reales."
        ),
    ])


# ═══════════════════════════════════════════════════════════════
# TAB: EDA
# ═══════════════════════════════════════════════════════════════
def make_eda():
    # PROMISE
    fig_promise_bal = go.Figure(go.Pie(
        labels=list(PROMISE_BALANCE.keys()), values=list(PROMISE_BALANCE.values()),
        marker=dict(colors=[NEON_GREEN, NEON_RED]),
        hole=0.5, textinfo="label+percent",
        textfont=dict(color="#fff", size=13),
    ))
    fig_promise_bal.update_layout(title="Balance de clases en PROMISE")
    dark_fig(fig_promise_bal, 380)

    corr = PROMISE_CK_CORR.sort_values("correlacion", ascending=True)
    fig_ck = go.Figure(go.Bar(
        x=corr["correlacion"], y=corr["metrica"], orientation="h",
        marker_color=[NEON_RED if v > 0 else NEON_CYAN for v in corr["correlacion"]],
        text=[f"{v:.3f}" for v in corr["correlacion"]], textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig_ck.update_layout(title="Correlacion metricas CK vs defectos")
    dark_fig(fig_ck, 550)

    # Red Hat cycle time
    ct = REDHAT_CYCLE_TIME
    fig_ct = go.Figure()
    for label, val, color in [("p50", ct["p50"], NEON_GREEN), ("p75", ct["p75"], NEON_AMBER), ("p90", ct["p90"], NEON_RED)]:
        fig_ct.add_trace(go.Bar(x=[label], y=[val], name=label, marker_color=color,
                                text=[f"{val:.0f}d"], textposition="outside",
                                textfont=dict(color=color, size=14, family="JetBrains Mono")))
    fig_ct.update_layout(title="Cycle Time — Red Hat JIRA (505K issues)", showlegend=False)
    dark_fig(fig_ct, 400)

    # GHArchive events
    evt = GHARCHIVE_EVENT_TYPES.copy()
    dora_types = {"PushEvent", "PullRequestEvent", "CreateEvent", "ReleaseEvent"}
    fig_gh = go.Figure(go.Bar(
        y=evt["tipo"][::-1], x=evt["conteo"][::-1], orientation="h",
        marker_color=[NEON_PURPLE if t in dora_types else "#4a5568" for t in evt["tipo"][::-1]],
        text=[f"{c:,}" for c in evt["conteo"][::-1]], textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig_gh.update_layout(title="Eventos GitHub — repos Apache 2022")
    dark_fig(fig_gh, 500)

    # Apache JIRA timeline
    fig_timeline = go.Figure(go.Bar(
        x=APACHE_ISSUES_POR_ANO["ano"], y=APACHE_ISSUES_POR_ANO["issues"],
        marker_color=NEON_CYAN,
        text=APACHE_ISSUES_POR_ANO["issues"], textposition="outside",
        textfont=dict(color=NEON_CYAN, size=11),
    ))
    fig_timeline.update_layout(title="Issues creados por año — Apache JIRA")
    dark_fig(fig_timeline, 380)

    # Defect density
    dd = PROMISE_DEFECT_DENSITY.sort_values("densidad", ascending=False).head(20)
    fig_dd = go.Figure(go.Bar(
        y=dd["proyecto"][::-1], x=dd["densidad"][::-1], orientation="h",
        marker_color=[NEON_RED if d > 0.5 else NEON_AMBER if d > 0.2 else NEON_GREEN
                      for d in dd["densidad"][::-1]],
        text=[f"{d:.0%}" for d in dd["densidad"][::-1]], textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig_dd.update_layout(title="Densidad de defectos por proyecto — Top 20")
    dark_fig(fig_dd, 550)

    # CFR donut
    cfr = GHARCHIVE_CFR
    fig_cfr = go.Figure(go.Pie(
        labels=["Merged", "Cerrado sin merge", "Otros"],
        values=[cfr["prs_merged"], cfr["prs_cerrados_sin_merge"],
                cfr["total_prs"] - cfr["prs_merged"] - cfr["prs_cerrados_sin_merge"]],
        marker=dict(colors=[NEON_GREEN, NEON_RED, "#4a5568"]),
        hole=0.5, textinfo="label+percent", textfont=dict(color="#fff", size=12),
    ))
    fig_cfr.update_layout(title=f"Pull Requests — Change Failure Rate proxy: {cfr['cfr_proxy']:.1%}")
    dark_fig(fig_cfr, 380)

    return html.Div([
        section("📊", "Exploracion de datos", "Los 4 EDAs que guiaron las decisiones de feature engineering"),

        # PROMISE
        html.Div([
            html.Div("PROMISE", className="card-title"),
            html.Div("144 CSVs con metricas CK de 41 proyectos open source. Cobertura con Apache JIRA: 1.6%.", className="card-text"),
        ], className="card"),
        html.Div([
            html.Div(dcc.Graph(figure=fig_promise_bal, config={"displayModeBar": False})),
            html.Div(dcc.Graph(figure=fig_ck, config={"displayModeBar": False})),
        ], className="grid-2"),
        html.Div(dcc.Graph(figure=fig_dd, config={"displayModeBar": False})),
        insight("RFC (Response for Class) es la metrica CK mas correlacionada con defectos (r=0.20). "
                "CAM tiene correlacion negativa — mas cohesion, menos bugs. "
                "xalan-2.7 tiene 98.8% de sus modulos con algun bug."),

        # Apache JIRA
        html.Div([
            html.Div("APACHE JIRA", className="card-title"),
            html.Div("978K issues de 42 proyectos. Define el target y las features de conteo. "
                      "70% de los sprints tiene al menos un defecto escapado — ese es el baseline a superar.", className="card-text"),
        ], className="card"),
        html.Div(dcc.Graph(figure=fig_timeline, config={"displayModeBar": False})),

        # Red Hat
        html.Div([
            html.Div("RED HAT JIRA", className="card-title"),
            html.Div(f"505K issues. Cycle time log-normal: p50={ct['p50']:.0f}d, p90={ct['p90']:.0f}d, max={ct['max']:,.0f}d. "
                      "Justifica log1p y flag cycle_missing.", className="card-text"),
        ], className="card"),
        html.Div(dcc.Graph(figure=fig_ct, config={"displayModeBar": False})),
        insight(f"La mitad de los issues se resuelven en menos de {ct['p50']:.0f} dias, pero el 10% mas lento "
                f"tarda mas de {ct['p90']:.0f} dias. El maximo es {ct['max']:,.0f} dias (~19 años)."),

        # GHArchive
        html.Div([
            html.Div("GHARCHIVE", className="card-title"),
            html.Div("Eventos GitHub 2022 de repos Apache. Deploy frequency y Change Failure Rate (proxies DORA). "
                      "Solo 9% de cobertura por mismatch temporal.", className="card-text"),
        ], className="card"),
        html.Div([
            html.Div(dcc.Graph(figure=fig_gh, config={"displayModeBar": False})),
            html.Div(dcc.Graph(figure=fig_cfr, config={"displayModeBar": False})),
        ], className="grid-2"),
        insight(f"En morado los eventos DORA. El proxy de CFR es {cfr['cfr_proxy']:.1%}: "
                f"de cada 100 PRs cerrados, {int(cfr['cfr_proxy']*100)} se descartaron sin merge."),

        # Summary
        html.Div([
            html.Div("LO QUE NOS LLEVAMOS AL MODELO", className="card-title"),
            html.Ul([
                html.Li("70% positivos → baseline trivial da F2=0.916. Ganar con precision, no recall."),
                html.Li("Drift temporal: cycle time bajo 96% entre 2008 y 2021."),
                html.Li("Cobertura cruzada baja: PROMISE 1.6%, DORA 9%. Flags de ausencia capturan esa señal."),
                html.Li("log1p obligatorio: cycle time p50=28d, max=7,030d."),
            ], style={"color": TEXT_MUTED, "lineHeight": "2", "paddingLeft": "20px"}),
        ], className="card card-glow"),
    ])

# ═══════════════════════════════════════════════════════════════
# TAB: RISK RADAR
# ═══════════════════════════════════════════════════════════════
def make_risk():
    snap = SNAP.copy()

    # Distribution
    fig_dist = go.Figure()
    for color, lo, hi, label in [
        (NEON_GREEN, 0, THRESHOLD, "Bajo"),
        (NEON_AMBER, THRESHOLD, 0.5, "Medio"),
        (NEON_RED, 0.5, 1.01, "Alto"),
    ]:
        vals = snap["probabilidad"][(snap["probabilidad"]>=lo)&(snap["probabilidad"]<hi)]
        fig_dist.add_trace(go.Histogram(
            x=vals, xbins=dict(start=lo, end=hi, size=0.02),
            marker_color=color, opacity=0.85, name=label,
        ))
    fig_dist.add_vline(x=THRESHOLD, line_dash="dash", line_color=NEON_PURPLE, line_width=2,
                       annotation=dict(text=f"Umbral {THRESHOLD}", font=dict(color=NEON_PURPLE, size=12),
                                       yanchor="bottom"))
    fig_dist.update_layout(barmode="stack", xaxis_title="Probabilidad estimada", yaxis_title="Sprints",
                           legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    dark_fig(fig_dist, 420)

    # Risk by project
    risk_proj = snap.groupby("project").agg(
        prob_media=("probabilidad", "mean"),
        n_alto=("riesgo_categoria", lambda x: (x=="alto").sum()),
        n_total=("riesgo_categoria", "count"),
    ).reset_index().sort_values("prob_media", ascending=False).head(15)

    fig_proj = go.Figure(go.Bar(
        y=risk_proj["project"][::-1], x=risk_proj["prob_media"][::-1], orientation="h",
        marker_color=[NEON_RED if p > 0.5 else NEON_AMBER if p > THRESHOLD else NEON_GREEN
                      for p in risk_proj["prob_media"][::-1]],
        text=[f"{p:.1%}" for p in risk_proj["prob_media"][::-1]],
        textposition="outside", textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig_proj.update_layout(title="Probabilidad promedio por proyecto — Top 15",
                           xaxis=dict(tickformat=".0%"))
    dark_fig(fig_proj, 500)

    # Risk by year
    risk_year = snap.groupby("sprint_year").agg(
        prob_media=("probabilidad", "mean"),
        n_alto=("riesgo_categoria", lambda x: (x=="alto").sum()),
        n=("probabilidad", "count"),
    ).reset_index()
    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(
        x=risk_year["sprint_year"], y=risk_year["n_alto"],
        name="Sprints alto riesgo", marker_color=NEON_RED, opacity=0.7,
    ))
    fig_year.add_trace(go.Scatter(
        x=risk_year["sprint_year"], y=risk_year["prob_media"],
        name="Prob. promedio", mode="lines+markers",
        line=dict(color=NEON_CYAN, width=3), marker=dict(size=10),
        yaxis="y2",
    ))
    fig_year.update_layout(
        title="Riesgo por año",
        yaxis=dict(title="Sprints alto riesgo"),
        yaxis2=dict(title="Prob. promedio", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    dark_fig(fig_year, 420)

    n_alto = (snap["riesgo_categoria"]=="alto").sum()
    n_medio = (snap["riesgo_categoria"]=="medio").sum()
    n_bajo = (snap["riesgo_categoria"]=="bajo").sum()

    return html.Div([
        section("🚦", "Radar de riesgo",
                f"{len(snap):,} sprints del periodo de evaluacion (2019-2021)"),

        html.Div([
            metric_card(f"{n_alto:,}", "ALTO RIESGO", f"{n_alto/len(snap)*100:.1f}%", NEON_RED),
            metric_card(f"{n_medio:,}", "MEDIO", f"{n_medio/len(snap)*100:.1f}%", NEON_AMBER),
            metric_card(f"{n_bajo:,}", "BAJO", f"{n_bajo/len(snap)*100:.1f}%", NEON_GREEN),
            metric_card(f"{THRESHOLD:.3f}", "UMBRAL", "punto de corte operativo", NEON_PURPLE),
        ], className="metrics-grid"),

        html.Div(dcc.Graph(figure=fig_dist, config={"displayModeBar": False})),
        insight("A la derecha del umbral estan los sprints que conviene revisar. "
                f"El modelo marca {n_alto+n_medio:,} sprints para atencion, cubriendo "
                f"la gran mayoria de los defectos reales."),

        html.Div([
            html.Div(dcc.Graph(figure=fig_proj, config={"displayModeBar": False})),
            html.Div(dcc.Graph(figure=fig_year, config={"displayModeBar": False})),
        ], className="grid-2"),

        # Top sprints table
        section("📋", "Sprints priorizados", "Los 30 sprints con mayor probabilidad de defecto escapado"),
        html.Div(id="risk-filters", children=[
            html.Div([
                html.Label("Proyecto", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                dcc.Dropdown(
                    id="risk-project-filter",
                    options=[{"label": "Todos", "value": "ALL"}] +
                            [{"label": p, "value": p} for p in sorted(snap["project"].unique())],
                    value="ALL", clearable=False,
                    style={"background": BG_ELEVATED, "color": TEXT_PRIMARY},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Año", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                dcc.Dropdown(
                    id="risk-year-filter",
                    options=[{"label": "Todos", "value": "ALL"}] +
                            [{"label": str(y), "value": y} for y in sorted(snap["sprint_year"].unique())],
                    value="ALL", clearable=False,
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Riesgo", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                dcc.Dropdown(
                    id="risk-level-filter",
                    options=[{"label": "Todos", "value": "ALL"},
                             {"label": "🔴 Alto", "value": "alto"},
                             {"label": "🟡 Medio", "value": "medio"},
                             {"label": "🟢 Bajo", "value": "bajo"}],
                    value="ALL", clearable=False,
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        html.Div(id="risk-table-container"),
    ])

# ═══════════════════════════════════════════════════════════════
# TAB: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
def make_explain():
    # SHAP global
    shap = SHAP_G.head(12).copy()
    shap["feature_label"] = shap["feature"].map(FEAT_DESC).fillna(shap["feature"])
    fig_shap = go.Figure(go.Bar(
        x=shap["importancia"][::-1], y=shap["feature_label"][::-1], orientation="h",
        marker=dict(
            color=shap["importancia"][::-1],
            colorscale=[[0, "#4a5568"], [0.5, NEON_PURPLE], [1, NEON_CYAN]],
        ),
        text=[f"{v:.3f}" for v in shap["importancia"][::-1]],
        textposition="outside", textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig_shap.update_layout(title="Importancia global de variables (SHAP)", showlegend=False)
    dark_fig(fig_shap, 500)

    # Reliability curve
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Calibracion perfecta",
                                  line=dict(color="#4a5568", dash="dash", width=2)))
    fig_rel.add_trace(go.Scatter(
        x=RELIABILITY["prob_predicha_media"], y=RELIABILITY["frecuencia_real"],
        mode="lines+markers", name="Modelo calibrado",
        line=dict(color=NEON_PURPLE, width=3),
        marker=dict(size=10, color=NEON_PURPLE, line=dict(width=2, color="#fff")),
    ))
    fig_rel.update_layout(title="Curva de confiabilidad — ¿las probabilidades son creibles?",
                          xaxis=dict(title="Probabilidad predicha", tickformat=".0%"),
                          yaxis=dict(title="Frecuencia real observada", tickformat=".0%"))
    dark_fig(fig_rel, 420)

    # Baseline comparison
    fig_comp = go.Figure()
    nombres = ["F2", "Precision", "Recall"]
    v_base  = [0.916, 0.704, 1.000]
    v_model = [MET["f2"], MET["precision"], MET["recall"]]
    fig_comp.add_trace(go.Bar(x=nombres, y=v_base, name="Baseline (siempre si)",
                              marker_color="#4a5568",
                              text=[f"{v:.3f}" for v in v_base], textposition="outside",
                              textfont=dict(color="#4a5568", size=13, family="JetBrains Mono")))
    fig_comp.add_trace(go.Bar(x=nombres, y=v_model, name="Champion XGBoost",
                              marker_color=NEON_PURPLE,
                              text=[f"{v:.3f}" for v in v_model], textposition="outside",
                              textfont=dict(color=NEON_PURPLE, size=13, family="JetBrains Mono")))
    fig_comp.update_layout(title="Champion vs Baseline — ¿el modelo aporta algo?",
                           barmode="group", yaxis=dict(range=[0, 1.15]),
                           legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    dark_fig(fig_comp, 420)

    return html.Div([
        section("🧠", "Inteligencia del modelo", "Como decide, por que es confiable, y que lo diferencia del azar"),

        html.Div([
            html.Div(dcc.Graph(figure=fig_shap, config={"displayModeBar": False})),
            html.Div([
                html.Div([
                    html.Div("TOP DRIVERS", className="card-title"),
                    *[html.Div([
                        html.Div(FEAT_DESC.get(r["feature"], r["feature"]),
                                 style={"fontWeight": "700", "color": "#fff", "fontSize": "14px"}),
                        html.Div(r.get("explicacion", ""), style={"color": TEXT_MUTED, "fontSize": "13px"}),
                    ], style={"borderBottom": f"1px solid {BORDER}", "padding": "12px 0"})
                      for _, r in shap.head(5).iterrows()],
                ], className="card"),
            ]),
        ], className="grid-2"),

        insight("Las tasas historicas por barrio dominan: el modelo aprende donde han ocurrido "
                "defectos antes y predice que seguiran ocurriendo. Las variables climaticas (en este "
                "caso, metricas DORA) aportan marginalmente por su baja cobertura (9%)."),

        html.Div([
            html.Div(dcc.Graph(figure=fig_rel, config={"displayModeBar": False})),
            html.Div(dcc.Graph(figure=fig_comp, config={"displayModeBar": False})),
        ], className="grid-2"),

        insight(f"El modelo mejora la precision de 70.4% (baseline) a {MET['precision']*100:.1f}% "
                f"manteniendo recall en {MET['recall']*100:.1f}%. "
                f"La calibracion isotonica reduce el Brier a {MET['brier']:.4f} — "
                f"cuando dice '30%', en la practica hay defectos en ~30 de cada 100 sprints."),
    ])

# ═══════════════════════════════════════════════════════════════
# TAB: HEALTH
# ═══════════════════════════════════════════════════════════════
def make_health():
    # F2 by year
    fig_f2 = go.Figure()
    fig_f2.add_hrect(y0=0.95, y1=1.0, fillcolor=NEON_GREEN, opacity=0.05, line_width=0)
    fig_f2.add_hrect(y0=0.90, y1=0.95, fillcolor=NEON_AMBER, opacity=0.05, line_width=0)
    fig_f2.add_hrect(y0=0.80, y1=0.90, fillcolor=NEON_RED, opacity=0.05, line_width=0)
    fig_f2.add_trace(go.Scatter(
        x=DRIFT_Y["año"], y=DRIFT_Y["f2"], mode="lines+markers+text",
        line=dict(color=NEON_PURPLE, width=3),
        marker=dict(size=12, color=NEON_PURPLE, line=dict(width=2, color="#fff")),
        text=[f"{v:.3f}" for v in DRIFT_Y["f2"]], textposition="top center",
        textfont=dict(color=NEON_PURPLE, size=12, family="JetBrains Mono"),
    ))
    fig_f2.add_hline(y=0.90, line_dash="dash", line_color=NEON_RED, line_width=1,
                     annotation=dict(text="Minimo aceptable", font=dict(color=NEON_RED, size=11)))
    fig_f2.update_layout(title="Evolucion del F2 por año",
                         yaxis=dict(title="F2-score", range=[0.80, 1.0]),
                         xaxis=dict(title="Año", dtick=1), showlegend=False)
    dark_fig(fig_f2, 420)

    # PSI by feature
    psi = DRIFT_PSI.head(10).copy()
    psi["feature_label"] = psi["feature"].map(FEAT_DESC).fillna(psi["feature"])
    fig_psi = go.Figure(go.Bar(
        x=psi["psi"][::-1], y=psi["feature_label"][::-1], orientation="h",
        marker_color=[NEON_RED if p > 0.25 else NEON_AMBER if p > 0.10 else NEON_GREEN
                      for p in psi["psi"][::-1]],
        text=[f"{p:.3f}" for p in psi["psi"][::-1]], textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=11),
    ))
    fig_psi.add_vline(x=0.10, line_dash="dot", line_color=NEON_AMBER, line_width=1)
    fig_psi.add_vline(x=0.25, line_dash="dash", line_color=NEON_RED, line_width=1)
    fig_psi.update_layout(title="Drift por variable (PSI: train vs test)")
    dark_fig(fig_psi, 450)

    psi_max = DRIFT_PSI["psi"].max()
    n_drift_alto = (DRIFT_PSI["psi"] > 0.25).sum()

    return html.Div([
        section("📡", "Salud del modelo", "Drift, degradacion y cuando reentrenar"),

        html.Div([
            metric_card("🟢" if MET["f2"] > 0.90 else "🟡" if MET["f2"] > 0.85 else "🔴",
                        "DESEMPEÑO", f"F2 = {MET['f2']:.3f}",
                        NEON_GREEN if MET["f2"] > 0.90 else NEON_AMBER),
            metric_card("🟢" if psi_max < 0.25 else "🟡" if psi_max < 0.40 else "🔴",
                        "DRIFT", f"PSI max = {psi_max:.3f}",
                        NEON_GREEN if psi_max < 0.25 else NEON_AMBER),
            metric_card("🟢" if MET["brier"] < 0.10 else "🟡",
                        "CALIBRACION", f"Brier = {MET['brier']:.4f}",
                        NEON_GREEN if MET["brier"] < 0.10 else NEON_AMBER),
            metric_card(f"{n_drift_alto}", "VARIABLES CON DRIFT ALTO",
                        "PSI > 0.25", NEON_RED if n_drift_alto > 0 else NEON_GREEN),
        ], className="metrics-grid"),

        html.Div([
            html.Div(dcc.Graph(figure=fig_f2, config={"displayModeBar": False})),
            html.Div(dcc.Graph(figure=fig_psi, config={"displayModeBar": False})),
        ], className="grid-2"),

        insight(
            "Las bandas verde/ambar/rojo en el grafico de F2 marcan las zonas de operacion. "
            "Si el F2 cae por debajo de 0.90 conviene reentrenar. "
            f"Actualmente hay {n_drift_alto} variable(s) con PSI > 0.25 (drift alto). "
            "Recomendacion: monitoreo mensual, reentrenamiento si el F2 baja 3+ puntos."
        ),
    ])

# ═══════════════════════════════════════════════════════════════
# TAB: PREDICTOR
# ═══════════════════════════════════════════════════════════════
def make_predictor():
    return html.Div([
        section("🔬", "Simulador de sprint", "Cambiar los numeros y ver como se mueve el riesgo"),
        html.Div([
            html.Div([
                html.Div("CONFIGURACION DEL SPRINT", className="card-title"),
                html.Div([
                    html.Div([
                        html.Label("Bugs", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                        dcc.Input(id="pred-bugs", type="number", value=5, min=0, max=100),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Stories", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                        dcc.Input(id="pred-stories", type="number", value=8, min=0, max=100),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Tasks", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                        dcc.Input(id="pred-tasks", type="number", value=10, min=0, max=100),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Cycle time (dias)", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                        dcc.Input(id="pred-cycle", type="number", value=45, min=0, max=3000),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "16px", "marginTop": "16px"}),
                html.Div([
                    html.Div([
                        html.Label("Año", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                        dcc.Input(id="pred-year", type="number", value=2021, min=2000, max=2025),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Mes", style={"color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "700"}),
                        dcc.Input(id="pred-month", type="number", value=6, min=1, max=12),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "16px", "marginTop": "12px"}),
            ], className="card"),

            html.Div(id="pred-output", className="card card-glow"),
        ], className="grid-2"),

        html.Div(id="pred-sensitivity"),
    ])


# ═══════════════════════════════════════════════════════════════
# APP LAYOUT
# ═══════════════════════════════════════════════════════════════
app = Dash(__name__, suppress_callback_exceptions=True,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "ShiftMetrics"
server = app.server

app.index_string = f'''<!DOCTYPE html>
<html><head>{{%metas%}}{{%favicon%}}{{%css%}}
<style>{CSS}</style>
</head><body>{{%app_entry%}}{{%config%}}{{%scripts%}}{{%renderer%}}</body></html>'''

app.layout = html.Div([
    html.Div([
        dcc.Tabs(id="main-tabs", value="home", className="custom-tabs", children=[
            dcc.Tab(label="⚡ Home", value="home", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="📊 Exploracion", value="eda", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🚦 Riesgo", value="risk", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🧠 Modelo", value="explain", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="📡 Salud", value="health", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="🔬 Simulador", value="predictor", className="tab", selected_className="tab--selected"),
        ]),
        html.Div(id="tab-content"),
        html.Div([
            f"ShiftMetrics · {MET.get('modelo_familia','')} · "
            f"Threshold {THRESHOLD:.3f} · "
            f"Evaluado: {MET.get('fecha_evaluacion','')}"
        ], className="footer"),
    ], className="page-shell"),
])


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "eda": return make_eda()
    if tab == "risk": return make_risk()
    if tab == "explain": return make_explain()
    if tab == "health": return make_health()
    if tab == "predictor": return make_predictor()
    return make_home()


@app.callback(
    Output("risk-table-container", "children"),
    Input("risk-project-filter", "value"),
    Input("risk-year-filter", "value"),
    Input("risk-level-filter", "value"),
)
def update_risk_table(project, year, level):
    df = SNAP.copy()
    if project != "ALL": df = df[df["project"] == project]
    if year != "ALL": df = df[df["sprint_year"] == year]
    if level != "ALL": df = df[df["riesgo_categoria"] == level]

    top = df.sort_values("probabilidad", ascending=False).head(30)
    top["prob_pct"] = (top["probabilidad"] * 100).round(1)
    top["driver1"] = top["driver_1"].map(FEAT_DESC).fillna(top["driver_1"])
    top["driver2"] = top["driver_2"].map(FEAT_DESC).fillna(top["driver_2"])
    top["semaforo"] = top["riesgo_categoria"].map({"alto":"🔴","medio":"🟡","bajo":"🟢"})

    return dash_table.DataTable(
        data=top[["semaforo","sprint_id","project","sprint_year","prob_pct",
                  "num_bugs_sprint","driver1","driver2"]].to_dict("records"),
        columns=[
            {"name":"","id":"semaforo"}, {"name":"Sprint","id":"sprint_id"},
            {"name":"Proyecto","id":"project"}, {"name":"Año","id":"sprint_year"},
            {"name":"Prob %","id":"prob_pct"}, {"name":"Bugs","id":"num_bugs_sprint"},
            {"name":"Driver 1","id":"driver1"}, {"name":"Driver 2","id":"driver2"},
        ],
        style_table={"overflowX": "auto"},
        page_size=15,
        sort_action="native",
        filter_action="native",
    )


@app.callback(
    Output("pred-output", "children"),
    Output("pred-sensitivity", "children"),
    Input("pred-bugs", "value"), Input("pred-stories", "value"),
    Input("pred-tasks", "value"), Input("pred-cycle", "value"),
    Input("pred-year", "value"), Input("pred-month", "value"),
)
def update_predictor(bugs, stories, tasks, cycle, year, month):
    bugs = int(bugs or 0); stories = int(stories or 0); tasks = int(tasks or 0)
    cycle = float(cycle or 0); year = int(year or 2021); month = int(month or 6)
    total = max(bugs + stories + tasks, 1)
    bsr = bugs / max(stories, 1) if stories > 0 else 0

    inputs = {
        "num_bugs_sprint": float(bugs), "num_stories_sprint": float(stories),
        "num_tasks_sprint": float(tasks), "total_issues_sprint": float(total),
        "log_avg_cycle_time": math.log1p(cycle),
        "log_bug_story_ratio": math.log1p(bsr) if stories > 0 else 0.0,
        "log_total_issues": math.log1p(total),
        "sprint_year": float(year),
        "sprint_month_sin": math.sin(2*math.pi*month/12),
        "sprint_month_cos": math.cos(2*math.pi*month/12),
        "deploy_frequency_weekly": 0.0, "change_failure_rate": 0.0,
        "bsr_missing": int(stories == 0), "cycle_missing": int(cycle == 0),
        "dora_missing": 1, "bugs_per_issue": bugs / total,
        "log_cycle_x_bsr": math.log1p(cycle) * (math.log1p(bsr) if stories > 0 else 0.0),
    }

    X = np.array([[inputs[f] for f in FEATURE_COLS]])
    model = CHAMPION["model"]
    prob = float(model.predict_proba(X)[0, 1])

    if prob >= 0.5: cat, color, emoji = "ALTO", NEON_RED, "🔴"
    elif prob >= THRESHOLD: cat, color, emoji = "MEDIO", NEON_AMBER, "🟡"
    else: cat, color, emoji = "BAJO", NEON_GREEN, "🟢"

    result = html.Div([
        html.Div("RESULTADO", style={"color": NEON_PURPLE, "fontSize": "11px",
                 "fontWeight": "700", "letterSpacing": "0.15em"}),
        html.Div(f"{prob*100:.1f}%", style={
            "fontFamily": "JetBrains Mono", "fontSize": "64px",
            "fontWeight": "900", "color": color, "lineHeight": "1", "marginTop": "8px",
        }),
        html.Div(f"{emoji} Riesgo {cat}", style={"fontSize": "18px", "color": color,
                 "fontWeight": "700", "marginTop": "8px"}),
        html.Div(f"Umbral operativo: {THRESHOLD:.3f}", style={
            "color": TEXT_MUTED, "fontSize": "13px", "marginTop": "12px"}),
    ], style={"textAlign": "center", "padding": "20px"})

    # Sensitivity: vary bugs
    bug_range = list(range(0, 21))
    probs_bugs = []
    for b in bug_range:
        inp2 = inputs.copy()
        inp2["num_bugs_sprint"] = float(b)
        t2 = max(b + stories + tasks, 1)
        inp2["total_issues_sprint"] = float(t2)
        inp2["log_total_issues"] = math.log1p(t2)
        inp2["bugs_per_issue"] = b / t2
        if stories > 0:
            bsr2 = b / stories
            inp2["log_bug_story_ratio"] = math.log1p(bsr2)
            inp2["log_cycle_x_bsr"] = inp2["log_avg_cycle_time"] * inp2["log_bug_story_ratio"]
        X2 = np.array([[inp2[f] for f in FEATURE_COLS]])
        probs_bugs.append(float(model.predict_proba(X2)[0, 1]))

    fig_sens = go.Figure()
    fig_sens.add_hrect(y0=0, y1=THRESHOLD, fillcolor=NEON_GREEN, opacity=0.05, line_width=0)
    fig_sens.add_hrect(y0=THRESHOLD, y1=0.5, fillcolor=NEON_AMBER, opacity=0.05, line_width=0)
    fig_sens.add_hrect(y0=0.5, y1=1, fillcolor=NEON_RED, opacity=0.05, line_width=0)
    fig_sens.add_trace(go.Scatter(
        x=bug_range, y=probs_bugs, mode="lines+markers",
        line=dict(color=NEON_PURPLE, width=3),
        marker=dict(size=8, color=NEON_PURPLE),
    ))
    fig_sens.add_hline(y=THRESHOLD, line_dash="dash", line_color=NEON_AMBER, line_width=1)
    # Mark current position
    fig_sens.add_trace(go.Scatter(
        x=[bugs], y=[prob], mode="markers",
        marker=dict(size=16, color=color, line=dict(width=3, color="#fff")),
        showlegend=False,
    ))
    fig_sens.update_layout(
        title="¿Como cambia el riesgo al variar los bugs?",
        xaxis_title="Numero de bugs en el sprint",
        yaxis=dict(title="Probabilidad de defecto", tickformat=".0%", range=[0, 1]),
        showlegend=False,
    )
    dark_fig(fig_sens, 400)

    sensitivity = html.Div([
        html.Div(dcc.Graph(figure=fig_sens, config={"displayModeBar": False})),
        insight(f"Con {bugs} bugs y un ciclo de {cycle:.0f} dias, "
                f"la probabilidad es {prob*100:.1f}%. "
                f"{'El sprint necesita atencion.' if prob >= THRESHOLD else 'El sprint esta en zona segura.'}"),
    ], className="card")

    return result, sensitivity


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
