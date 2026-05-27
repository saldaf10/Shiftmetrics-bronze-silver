"""ShiftMetrics SI7009 dashboard.

Dash app for the visualization layer of the project.
The dashboard unifies pipeline status, model comparison, calibration,
drift, SHAP evidence, and an interactive prediction sandbox.
"""

from __future__ import annotations

import base64
import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table

from config import FEATURE_COLS, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME, RANDOM_SEED


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "docs" / "assets"
OPERATING_THRESHOLD = 0.220
MODEL_VERSION = "v4"
BUILD_DATE = date.today().isoformat()


SUMMARY = {
    "experiment": MLFLOW_EXPERIMENT,
    "registry": MODEL_REGISTRY_NAME,
    "version": MODEL_VERSION,
    "mlflow": MLFLOW_TRACKING_URI,
}

PIPELINE_STATUS = pd.DataFrame(
    [
        {"layer": "Bronze", "status": "Complete", "detail": "Raw project sources and extract jobs"},
        {"layer": "Silver", "status": "Complete", "detail": "Cleaned outputs validated in notebooks"},
        {"layer": "Gold", "status": "Ready", "detail": "Feature table for sprint scoring"},
        {"layer": "ML", "status": "Complete", "detail": "XGBoost, calibration, SHAP, drift"},
        {"layer": "Viz", "status": "Ready", "detail": "Dash UI with design lab, concepts and sandbox"},
    ]
)

EXECUTIVE_KPIS = pd.DataFrame(
    [
        {"kpi": "Champion test F2", "value": "0.9655", "interpretation": "El modelo captura casi todos los sprints de alto riesgo."},
        {"kpi": "Recall test", "value": "0.9953", "interpretation": "Muy pocos casos de alto riesgo quedan fuera del radar."},
        {"kpi": "Precision test", "value": "0.8623", "interpretation": "Las alertas tienen señal suficiente para priorizar revisión."},
        {"kpi": "Brier test", "value": "0.0666", "interpretation": "Las probabilidades ya son utilizables para priorización."},
    ]
)

EXECUTIVE_IMPACT = pd.DataFrame(
    [
        {"mensaje": "Umbral operativo", "valor": "0.220", "uso": "Punto de decisión recomendado para revisión de sprint."},
        {"mensaje": "Flagging rate", "valor": "74.2%", "uso": "Usar como priorizador, no como bloqueo binario."},
        {"mensaje": "Generalización temporal", "valor": "ΔF2 = -0.43 pp", "uso": "La pérdida entre val y test es estable y controlada."},
        {"mensaje": "Estabilidad cross-project", "valor": "LOPO mean F2 = 0.9546", "uso": "El desempeño se mantiene al mover el modelo entre proyectos."},
    ]
)

EXECUTIVE_ACTIONS = [
    "Priorizar QA y revisión de alcance en sprints con score alto.",
    "Monitorear drift y recalibrar si el Brier supera 0.10.",
    "Usar SHAP para explicar por qué un sprint quedó en zona roja.",
    "No usar el modelo para evaluar personas; el objeto es el sprint.",
]

MODEL_LEADERBOARD = pd.DataFrame(
    [
        {"modelo": "Baseline", "f2": 0.916, "pr_auc": 0.686, "precision": 0.686, "recall": 1.000, "brier": 0.217},
        {"modelo": "Logistic regression", "f2": 0.861, "pr_auc": 0.977, "precision": 0.959, "recall": 0.839, "brier": 0.100},
        {"modelo": "XGBoost", "f2": 0.969, "pr_auc": 0.984, "precision": 0.875, "recall": 0.998, "brier": 0.106},
        {"modelo": "Champion calibrated", "f2": 0.970, "pr_auc": 0.984, "precision": 0.873, "recall": 0.997, "brier": 0.061},
    ]
)

CALIBRATION_ROWS = [
    {"split": "Cal (2015)", "raw": 0.0973, "isotonic": 0.0530, "reduction": "45.5%"},
    {"split": "Val (2016-2018)", "raw": 0.1062, "isotonic": 0.0605, "reduction": "43.0%"},
    {"split": "Test (2019-2021)", "raw": 0.1137, "isotonic": 0.0666, "reduction": "41.4%"},
]

DRIFT_ROWS = pd.DataFrame(
    [
        {"año": 2000, "f2": 0.918, "psi": 0.04},
        {"año": 2008, "f2": 0.942, "psi": 0.08},
        {"año": 2014, "f2": 0.960, "psi": 0.11},
        {"año": 2018, "f2": 0.969, "psi": 0.14},
        {"año": 2021, "f2": 0.902, "psi": 0.21},
    ]
)

SHAP_FEATURES = pd.DataFrame(
    [
        {"feature": "num_bugs_sprint", "importance": 1.00, "reason": "Más bugs elevan el riesgo de escape."},
        {"feature": "log_avg_cycle_time", "importance": 0.82, "reason": "Ciclos largos aumentan la ventana de exposición."},
        {"feature": "bugs_per_issue", "importance": 0.69, "reason": "La densidad de bugs captura carga correctiva."},
        {"feature": "cycle_missing", "importance": 0.58, "reason": "La ausencia de ciclo también aporta señal."},
        {"feature": "total_issues_sprint", "importance": 0.47, "reason": "Sprints grandes amplían la superficie de error."},
    ]
)

DESIGN_PILLARS = pd.DataFrame(
    [
        {"principio": "Claridad", "detalle": "El mensaje debe ser inmediato y sin ambigüedades."},
        {"principio": "Precisión", "detalle": "No distorsionar proporciones, escalas ni unidades."},
        {"principio": "Eficiencia", "detalle": "Maximizar información útil y minimizar ruido."},
        {"principio": "Contexto", "detalle": "Incluir siempre fuente, título, unidades y fecha."},
    ]
)

VARIABLE_GUIDE = pd.DataFrame(
    [
        {"tipo": "Nominal", "pregunta": "Comparar categorías", "grafica": "Barras horizontales / Treemap", "evitar": "Línea de tiempo"},
        {"tipo": "Ordinal", "pregunta": "Mostrar jerarquía", "grafica": "Barras ordenadas / Heatmap", "evitar": "Pie sin orden"},
        {"tipo": "Discreta", "pregunta": "Distribución de conteos", "grafica": "Barras / Lollipop / Boxplot", "evitar": "Histograma continuo"},
        {"tipo": "Continua", "pregunta": "Distribución y tendencia", "grafica": "Histograma / Violin / Línea", "evitar": "Pie chart"},
    ]
)

LINEAR_STORY_DATA = pd.DataFrame(
    [
        {"periodo": "Q1", "Producto A": 120, "Producto B": 140, "Producto C": 160},
        {"periodo": "Q2", "Producto A": 110, "Producto B": 150, "Producto C": 185},
        {"periodo": "Q3", "Producto A": 105, "Producto B": 155, "Producto C": 220},
        {"periodo": "Q4", "Producto A": 100, "Producto B": 148, "Producto C": 250},
    ]
)

HIPPO_POINTS = [
    "Parálisis por jerarquía: el equipo busca aprobación y no verdad.",
    "Pérdida de talento: los expertos dejan de ser escuchados.",
    "Riesgo estratégico: se ignoran señales tempranas que solo muestran los datos.",
]

NARRATIVE_CONTEXT = pd.DataFrame(
    [
        {"etapa": "Contexto", "objetivo": "Explicar el problema y la audiencia."},
        {"etapa": "Datos", "objetivo": "Elegir la variable correcta y la unidad correcta."},
        {"etapa": "Contraste", "objetivo": "Resaltar el patrón o cambio que importa."},
        {"etapa": "Mensaje", "objetivo": "Cerrar con una decisión accionable."},
    ]
)

PREDICTOR_DEFAULTS = {
    "num_bugs_sprint": 6,
    "num_stories_sprint": 5,
    "num_tasks_sprint": 7,
    "avg_cycle_time_days": 35,
    "sprint_year": 2019,
    "sprint_month": 6,
    "deploy_frequency_weekly": 1.8,
    "change_failure_rate": 0.26,
    "cycle_missing": 0,
}

VISUAL_DEFAULTS = {
    "chart_kind": "line",
    "theme": "white",
    "audience": "manager",
    "variable_type": "Continua",
}

EXECUTIVE_DEFAULTS = {
    "highlight": "champion",
}

CHAMPION_URI_CANDIDATES = [
    f"models:/{MODEL_REGISTRY_NAME}@champion",
    f"models:/{MODEL_REGISTRY_NAME}@production",
    f"models:/{MODEL_REGISTRY_NAME}/{MODEL_VERSION.lstrip('v')}",
]

CANONICAL_MLFLOW_RUNS = {
    "champion_selection": "b81e450774924dfc94cb82bd173cc2ef",
    "calibration": "355c43bed63b4c7e96d9afd37eacbd4b",
    "lopo_cv": "cc338b317d73472c96cfaa1b9f6d0a86",
    "shap": "0e6c9b9597cf4023a0b9a1976fd6c05f",
    "drift": "d536b2fdbac74a798ad0a2e8b5081b4e",
    "threshold": "4f398f29acdb4d92b5710aef391b08ed",
    "final_eval": "90d0407fefeb449a8b99ea3c6dc3f8dc",
}


def load_png_b64(name: str) -> str:
    path = ASSET_DIR / name
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def pill(text: str, variant: str = "neutral") -> html.Span:
    return html.Span(text, className=f"pill pill-{variant}")


def metric_card(title: str, value: str, subtitle: str) -> html.Div:
    return html.Div(
        [html.Div(title, className="metric-title"), html.Div(value, className="metric-value"), html.Div(subtitle, className="metric-subtitle")],
        className="metric-card glass-card",
    )


def image_card(title: str, image_name: str, description: str) -> html.Div:
    body = [html.H4(title, className="card-title"), html.P(description, className="card-subtitle")]
    encoded = load_png_b64(image_name)
    if encoded:
        body.append(html.Img(src=f"data:image/png;base64,{encoded}", className="dashboard-image"))
    else:
        body.append(html.Div("Imagen no disponible en el repositorio", className="image-placeholder"))
    return html.Div(body, className="glass-card image-card")


def pipeline_table() -> html.Table:
    header = html.Thead(html.Tr([html.Th("Capa"), html.Th("Estado"), html.Th("Detalle")]))
    rows = []
    for row in PIPELINE_STATUS.itertuples(index=False):
        rows.append(
            html.Tr(
                [
                    html.Td(row.layer),
                    html.Td(pill(row.status, "ready" if row.status in {"Complete", "Ready"} else "progress")),
                    html.Td(row.detail),
                ]
            )
        )
    return html.Table([header, html.Tbody(rows)], className="status-table")


def build_leaderboard_figure() -> go.Figure:
    df = MODEL_LEADERBOARD.melt(id_vars="modelo", value_vars=["f2", "pr_auc"], var_name="métrica", value_name="score")
    fig = px.bar(df, x="modelo", y="score", color="métrica", barmode="group", color_discrete_map={"f2": "#22d3ee", "pr_auc": "#14b8a6"})
    fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(range=[0, 1.05], title_text="Score")
    return fig


def build_calibration_figure() -> go.Figure:
    df = pd.DataFrame(CALIBRATION_ROWS).melt(id_vars="split", var_name="tipo", value_name="brier")
    fig = px.bar(df, x="split", y="brier", color="tipo", barmode="group", color_discrete_map={"raw": "#fb7185", "isotonic": "#22d3ee"})
    fig.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def build_drift_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DRIFT_ROWS["año"], y=DRIFT_ROWS["f2"], mode="lines+markers", name="F2", line=dict(color="#22d3ee", width=3)))
    fig.add_trace(go.Bar(x=DRIFT_ROWS["año"], y=DRIFT_ROWS["psi"], name="PSI", marker_color="#f59e0b", opacity=0.65, yaxis="y2"))
    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(title="F2", range=[0.85, 1.0]),
        yaxis2=dict(title="PSI", overlaying="y", side="right", range=[0, 0.3]),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig


def build_shap_figure() -> go.Figure:
    fig = px.bar(SHAP_FEATURES.sort_values("importance"), x="importance", y="feature", orientation="h")
    fig.update_layout(template="plotly_dark", height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(title_text="Importancia relativa")
    return fig


@lru_cache(maxsize=1)
def load_executive_ml_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "source": "fallback",
        "error": "No se pudo cargar MLflow; usando valores documentados.",
        "champion_name": "XGBoost",
        "calibration_method": "isotonic",
        "threshold_f2opt": OPERATING_THRESHOLD,
        "threshold_roi": OPERATING_THRESHOLD,
        "val_f2": 0.9698,
        "val_brier": 0.0605,
        "test_f2": 0.9655,
        "test_brier": 0.0666,
        "test_recall": 0.9953,
        "test_precision": 0.8623,
        "test_pr_auc": 0.9791,
        "flagging_rate_test": 0.7420,
        "lopo_f2_mean": 0.9526,
        "lopo_f2_std": 0.0390,
        "lopo_cal_f2_mean": 0.9546,
        "lopo_cal_f2_std": 0.0370,
        "test_f2_ci_lo": 0.9627,
        "test_f2_ci_hi": 0.9684,
        "test_recall_ci_lo": 0.9932,
        "test_recall_ci_hi": 0.9972,
    }

    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run = client.get_run(CANONICAL_MLFLOW_RUNS["final_eval"])
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        def metric(name: str, default: float) -> float:
            value = metrics.get(name)
            return float(value) if value is not None else default

        evidence.update(
            {
                "source": "mlflow",
                "error": "",
                "champion_name": tags.get("champion_name", evidence["champion_name"]),
                "calibration_method": params.get("calibration_method", evidence["calibration_method"]),
                "threshold_f2opt": float(params.get("threshold_f2opt", evidence["threshold_f2opt"])),
                "threshold_roi": float(params.get("threshold_roi", evidence["threshold_roi"])),
                "val_f2": metric("val_f2", evidence["val_f2"]),
                "val_brier": metric("val_brier", evidence["val_brier"]),
                "test_f2": metric("test_f2", evidence["test_f2"]),
                "test_brier": metric("test_brier", evidence["test_brier"]),
                "test_recall": metric("test_recall", evidence["test_recall"]),
                "test_precision": metric("test_precision", evidence["test_precision"]),
                "test_pr_auc": metric("test_pr_auc", evidence["test_pr_auc"]),
                "flagging_rate_test": metric("flagging_rate_test", evidence["flagging_rate_test"]),
                "lopo_f2_mean": metric("lopo_f2_mean", evidence["lopo_f2_mean"]),
                "lopo_f2_std": metric("lopo_f2_std", evidence["lopo_f2_std"]),
                "lopo_cal_f2_mean": metric("lopo_cal_f2_mean", evidence["lopo_cal_f2_mean"]),
                "lopo_cal_f2_std": metric("lopo_cal_f2_std", evidence["lopo_cal_f2_std"]),
                "test_f2_ci_lo": metric("test_f2_ci_lo", evidence["test_f2_ci_lo"]),
                "test_f2_ci_hi": metric("test_f2_ci_hi", evidence["test_f2_ci_hi"]),
                "test_recall_ci_lo": metric("test_recall_ci_lo", evidence["test_recall_ci_lo"]),
                "test_recall_ci_hi": metric("test_recall_ci_hi", evidence["test_recall_ci_hi"]),
                "run_id": run.info.run_id,
            }
        )

        summary_path = Path("/tmp/pipeline_summary.json")
        if summary_path.exists():
            try:
                evidence["pipeline_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                evidence["pipeline_summary"] = {}
        else:
            evidence["pipeline_summary"] = {}
    except Exception as exc:
        evidence["error"] = f"{exc}"

    return evidence


def build_executive_figure(evidence: dict[str, Any]) -> go.Figure:
    champion_f2 = float(evidence.get("test_f2", 0.9655))
    champion_precision = float(evidence.get("test_precision", 0.8623))
    champion_brier = float(evidence.get("test_brier", 0.0666))
    df = pd.DataFrame(
        [
            {"modelo": "Baseline", "métrica": "F2", "score": 0.916},
            {"modelo": "Baseline", "métrica": "Precision", "score": 0.686},
            {"modelo": "Baseline", "métrica": "Brier", "score": 0.217},
            {"modelo": "Champion", "métrica": "F2", "score": champion_f2},
            {"modelo": "Champion", "métrica": "Precision", "score": champion_precision},
            {"modelo": "Champion", "métrica": "Brier", "score": champion_brier},
        ]
    )
    fig = px.bar(df, x="métrica", y="score", color="modelo", barmode="group", color_discrete_map={"Baseline": "#94a3b8", "Champion": "#22d3ee"})
    fig.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(range=[0, 1.05], title_text="Valor")
    return fig


def build_design_theme_figure(theme: str) -> go.Figure:
    story = LINEAR_STORY_DATA.melt(id_vars="periodo", var_name="producto", value_name="ventas")
    chart_colors = {"Producto A": "#94a3b8", "Producto B": "#f59e0b", "Producto C": "#22d3ee"}
    dark = theme == "dark"
    template = "plotly_dark" if dark else "plotly_white"
    paper = "rgba(0,0,0,0)" if dark else "#ffffff"
    plot = "rgba(0,0,0,0)" if dark else "#ffffff"
    fig = px.line(story, x="periodo", y="ventas", color="producto", markers=True, color_discrete_map=chart_colors)
    fig.update_layout(template=template, height=360, paper_bgcolor=paper, plot_bgcolor=plot, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Producto")
    fig.update_yaxes(title_text="Ventas totales $")
    return fig


def build_design_bar_figure(theme: str) -> go.Figure:
    story = LINEAR_STORY_DATA.melt(id_vars="periodo", var_name="producto", value_name="ventas")
    chart_colors = {"Producto A": "#94a3b8", "Producto B": "#f59e0b", "Producto C": "#22d3ee"}
    dark = theme == "dark"
    template = "plotly_dark" if dark else "plotly_white"
    paper = "rgba(0,0,0,0)" if dark else "#ffffff"
    plot = "rgba(0,0,0,0)" if dark else "#ffffff"
    fig = px.bar(story, x="periodo", y="ventas", color="producto", barmode="group", color_discrete_map=chart_colors)
    fig.update_layout(template=template, height=360, paper_bgcolor=paper, plot_bgcolor=plot, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Producto")
    fig.update_yaxes(title_text="Ventas totales $")
    return fig


def build_design_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        data=LINEAR_STORY_DATA.to_dict("records"),
        columns=[{"name": c, "id": c} for c in LINEAR_STORY_DATA.columns],
        style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
        style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
    )


def audience_message(audience: str, chart_kind: str) -> str:
    recommendations = {
        "executive": "Para una audiencia ejecutiva, una línea limpia con el mensaje central y pocas series es más persuasiva.",
        "manager": "Para un gerente, la barra o la línea funcionan, pero la línea comunica tendencia con más rapidez.",
        "technical": "Para una audiencia técnica, la tabla sirve para validación; el gráfico sigue siendo mejor para lectura rápida.",
    }
    chosen = recommendations.get(audience, recommendations["manager"])
    if chart_kind == "table":
        chosen += " Sin embargo, la tabla es útil como respaldo, no como primera vista."
    elif chart_kind == "bar":
        chosen += " La barra ayuda a comparar niveles, pero la tendencia se lee más rápido en línea."
    else:
        chosen += ""
    return chosen


def variable_recommendation(variable_type: str) -> str:
    row = VARIABLE_GUIDE.loc[VARIABLE_GUIDE["tipo"].str.lower() == variable_type.lower()]
    if row.empty:
        return "Selecciona un tipo de variable para obtener una recomendación." 
    item = row.iloc[0]
    return f"Para variables {item['tipo']}: {item['grafica']} para responder '{item['pregunta']}'. Evita {item['evitar']}."


def make_design_lab_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Contexto + Datos + Contraste", className="section-title"),
                            html.P("El dashboard debe decidir rápido qué historia contar, a quién y con qué nivel de detalle.", className="body-copy"),
                            html.Div([pill("Clarity", "ready"), pill("Precision", "neutral"), pill("Efficiency", "neutral"), pill("Context", "neutral")], className="pill-row"),
                        ],
                        className="glass-card section-card",
                    ),
                    html.Div(
                        [
                            html.H3("HiPPO: el antídoto visual", className="section-title"),
                            html.Ul([html.Li(point) for point in HIPPO_POINTS], className="clean-list"),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="chart-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Diseño narrativo", className="section-title"),
                            html.P("Mismos datos, distintos mensajes. La interacción te deja comparar tabla, barra y línea con la misma información.", className="body-copy"),
                            html.Div(
                                [
                                    html.Div([html.Label("Audiencia"), dcc.Dropdown(id="viz-audience", options=[{"label": "Ejecutiva", "value": "executive"}, {"label": "Gerencial", "value": "manager"}, {"label": "Técnica", "value": "technical"}], value=VISUAL_DEFAULTS["audience"], clearable=False)], className="control-block"),
                                    html.Div([html.Label("Tipo de vista"), dcc.RadioItems(id="viz-chart-kind", options=[{"label": "Línea", "value": "line"}, {"label": "Barras", "value": "bar"}, {"label": "Tabla", "value": "table"}], value=VISUAL_DEFAULTS["chart_kind"], inline=True, className="checklist")], className="control-block"),
                                    html.Div([html.Label("Fondo"), dcc.RadioItems(id="viz-theme", options=[{"label": "Claro", "value": "white"}, {"label": "Oscuro", "value": "dark"}], value=VISUAL_DEFAULTS["theme"], inline=True, className="checklist")], className="control-block"),
                                ],
                                className="predictor-controls",
                            ),
                        ],
                        className="glass-card section-card",
                    ),
                    html.Div(
                        [
                            html.H3("Elección de gráfica según variable", className="section-title"),
                            html.P("El tipo de variable determina el lenguaje visual más apropiado.", className="body-copy"),
                            html.Div([html.Label("Tipo de variable"), dcc.Dropdown(id="viz-variable-type", options=[{"label": row, "value": row} for row in VARIABLE_GUIDE["tipo"]], value=VISUAL_DEFAULTS["variable_type"], clearable=False)], className="control-block"),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="two-column",
            ),
            html.Div(id="viz-advice-box", className="glass-card section-card"),
            html.Div(id="viz-variable-box", className="glass-card section-card"),
            html.Div(id="viz-story-content", className="glass-card chart-card"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Ciclo cognitivo", className="section-title"),
                            html.Ul([html.Li("Memoria sensorial: captar el patrón en < 1s."), html.Li("Atención de trabajo: no saturar con más de 4-7 elementos."), html.Li("Memoria de largo plazo: convertir datos en criterio." )], className="clean-list"),
                        ],
                        className="glass-card section-card",
                    ),
                    html.Div(
                        [
                            html.H3("Resumen del argumento", className="section-title"),
                            dash_table.DataTable(
                                data=NARRATIVE_CONTEXT.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in NARRATIVE_CONTEXT.columns],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="two-column",
            ),
        ],
        className="section-stack",
    )


def make_concepts_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Principios fundamentales", className="section-title"),
                            dash_table.DataTable(
                                data=DESIGN_PILLARS.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in DESIGN_PILLARS.columns],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card table-card",
                    ),
                    html.Div(
                        [
                            html.H3("Gobernanza visual", className="section-title"),
                            html.Ul(
                                [
                                    html.Li("Cada figura debe incluir título, fuente, fecha y unidad de medida."),
                                    html.Li("No mezclar objetivos exploratorios con mensajes ejecutivos en la misma vista."),
                                    html.Li("Las escalas deben ser honestas y comparables."),
                                    html.Li("Los gráficos son argumentos; cada uno debe responder una pregunta."),
                                ],
                                className="clean-list",
                            ),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="chart-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Gestalt y carga cognitiva", className="section-title"),
                            html.P("Proximidad, similitud, cierre y contraste ordenan la percepción antes de cualquier lectura consciente.", className="body-copy"),
                            html.Div([pill("Proximidad", "ready"), pill("Similitud", "neutral"), pill("Cierre", "neutral"), pill("Contraste", "neutral")], className="pill-row"),
                        ],
                        className="glass-card section-card",
                    ),
                    html.Div(
                        [
                            html.H3("Exploratorio vs aclaratorio", className="section-title"),
                            html.P("Explorar significa descubrir; aclarar significa convencer. La UI separa ambas intenciones.", className="body-copy"),
                            html.Div([pill("Exploratorio", "neutral"), pill("Aclaratorio", "ready"), pill("Argumento visual", "danger")], className="pill-row"),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="two-column",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Taxonomía de variables", className="section-title"),
                            dash_table.DataTable(
                                data=VARIABLE_GUIDE.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in VARIABLE_GUIDE.columns],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card table-card",
                    ),
                    html.Div(
                        [
                            html.H3("Distribuciones y formato", className="section-title"),
                            html.Ul(
                                [
                                    html.Li("Histograma + boxplot para una primera lectura univariada."),
                                    html.Li("Scatter + línea de tendencia para bivariado numérico."),
                                    html.Li("Tablas solo cuando el usuario necesite validación exacta."),
                                    html.Li("La paleta y el fondo deben reforzar, no competir, con el mensaje."),
                                ],
                                className="clean-list",
                            ),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="chart-grid",
            ),
        ],
        className="section-stack",
    )


def build_story_content(chart_kind: str, theme: str) -> html.Div:
    if chart_kind == "table":
        return html.Div(
            [
                html.H3("Mismos datos en tabla", className="section-title"),
                html.P("La tabla valida valores exactos, pero oculta la tendencia con más esfuerzo cognitivo.", className="body-copy"),
                build_design_table(),
            ],
            className="story-panel",
        )

    if chart_kind == "bar":
        figure = build_design_bar_figure(theme)
        caption = "Barras comparan bien magnitudes, pero introducen más ruido visual que una línea para la tendencia." 
    else:
        figure = build_design_theme_figure(theme)
        caption = "La línea comunica el liderazgo de Producto C y la tendencia general en segundos."

    return html.Div(
        [
            html.H3("Mismos datos, distinto mensaje", className="section-title"),
            html.P(caption, className="body-copy"),
            dcc.Graph(figure=figure, className="chart"),
        ],
        className="story-panel",
    )


def build_story_advice(audience: str, chart_kind: str) -> html.Div:
    return html.Div(
        [
            html.H3("Recomendación narrativa", className="section-title"),
            html.P(audience_message(audience, chart_kind), className="body-copy"),
        ],
        className="advice-box",
    )


def build_variable_advice(variable_type: str) -> html.Div:
    return html.Div(
        [
            html.H3("Regla de selección", className="section-title"),
            html.P(variable_recommendation(variable_type), className="body-copy"),
        ],
        className="advice-box",
    )


@lru_cache(maxsize=1)
def build_runtime_summary() -> tuple[str, str, str, Any | None]:
    try:
        import mlflow.sklearn  # type: ignore
    except Exception as exc:
        return "offline", "Fallback heurístico", f"MLflow no disponible: {exc}", None

    last_error = None
    for uri in CHAMPION_URI_CANDIDATES:
        try:
            model = mlflow.sklearn.load_model(uri)
            return "live", f"Champion live · {uri}", "Modelo cargado desde el registry", model
        except Exception as exc:
            last_error = str(exc)
    return "offline", "Fallback heurístico", last_error or "No se pudo cargar el champion", None


def make_footer() -> html.Footer:
    return html.Footer(
        [
            html.Div(f"Experimento: {SUMMARY['experiment']}"),
            html.Div(f"Registry: {SUMMARY['registry']} ({SUMMARY['version']})"),
            html.Div(f"Threshold operativo: {OPERATING_THRESHOLD:.3f}"),
            html.Div(f"Build: {BUILD_DATE} · Seed: {RANDOM_SEED}"),
        ],
        className="dashboard-footer glass-card",
    )


def make_overview() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    metric_card("F2 validación", "0.970", "Champion XGBoost + isotónica"),
                    metric_card("F2 test", "0.9655", "IC bootstrap [0.9627, 0.9684]"),
                    metric_card("Recall test", "0.9953", "Split 2019-2021"),
                    metric_card("Brier test", "0.0666", "Calibración isotónica"),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    html.Div([html.H3("Estado del pipeline", className="section-title"), pipeline_table()], className="glass-card section-card"),
                    html.Div(
                        [
                            html.H3("Contexto operativo", className="section-title"),
                            html.P("El dashboard resume el estado del sistema de punta a punta y separa validación y test.", className="body-copy"),
                            html.Div([pill("Live MLflow", "ready"), pill("Threshold 0.220", "neutral"), pill("Registry v4", "neutral")], className="pill-row"),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="two-column",
            ),
            html.Div(
                [dcc.Graph(figure=build_leaderboard_figure(), className="chart"), dcc.Graph(figure=build_calibration_figure(), className="chart")],
                className="chart-grid",
            ),
        ],
        className="section-stack",
    )


def make_executive_tab() -> html.Div:
    evidence = load_executive_ml_evidence()
    champion_name = evidence.get("champion_name", "XGBoost")
    calibration_method = evidence.get("calibration_method", "isotonic")
    source_label = "MLflow" if evidence.get("source") == "mlflow" else "documentación"
    generalization_gap = float(evidence.get("val_f2", 0.9698)) - float(evidence.get("test_f2", 0.9655))
    flagging_rate = float(evidence.get("flagging_rate_test", 0.7420))
    threshold_f2 = float(evidence.get("threshold_f2opt", OPERATING_THRESHOLD))
    return html.Div(
        [
            html.Div(
                [
                    metric_card("Test F2", f"{float(evidence.get('test_f2', 0.9655)):.4f}", f"Champion {champion_name} · {calibration_method}"),
                    metric_card("Test precision", f"{float(evidence.get('test_precision', 0.8623)):.4f}", "Señal suficiente para priorizar revisión"),
                    metric_card("Recall", f"{float(evidence.get('test_recall', 0.9953)):.4f}", "Casi todos los sprints de riesgo quedan marcados"),
                    metric_card("Threshold", f"{threshold_f2:.3f}", "Punto operativo recomendado"),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Decisión ejecutiva", className="section-title"),
                            html.P(
                                f"El champion {champion_name} ya es útil para priorizar QA y revisión de sprint. No reemplaza la decisión humana: la acelera, la ordena y la vuelve trazable. La calibración {calibration_method} y el threshold {threshold_f2:.3f} sustentan la decisión.",
                                className="body-copy",
                            ),
                            html.Div([pill("Priorizar revisión", "ready"), pill("Monitorear drift", "neutral"), pill(f"Fuente {source_label}", "neutral")], className="pill-row"),
                        ],
                        className="glass-card section-card",
                    ),
                    html.Div(
                        [
                            html.H3("Estado del modelo", className="section-title"),
                            html.Ul(
                                [
                                    html.Li(f"Champion calibrado con {calibration_method} y threshold {threshold_f2:.3f}."),
                                    html.Li(f"La brecha val/test de F2 es {generalization_gap:+.4f}, consistente con drift controlado."),
                                    html.Li(f"El flagging rate en test es {flagging_rate:.1%}, útil para priorización y no para bloqueo binario."),
                                ],
                                className="clean-list",
                            ),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="two-column",
            ),
            html.Div(
                [
                    dcc.Graph(figure=build_executive_figure(evidence), className="chart"),
                    html.Div(
                        [
                            html.H3("Qué cambia con el champion", className="section-title"),
                            dash_table.DataTable(
                                data=[
                                    {
                                        "mensaje": "Umbral operativo",
                                        "valor": f"{threshold_f2:.3f}",
                                        "uso": "Punto de decisión recomendado para revisión de sprint.",
                                    },
                                    {
                                        "mensaje": "Flagging rate",
                                        "valor": f"{flagging_rate:.1%}",
                                        "uso": "Usar como priorizador, no como bloqueo binario.",
                                    },
                                    {
                                        "mensaje": "Generalización temporal",
                                        "valor": f"ΔF2 = {generalization_gap:+.2%}",
                                        "uso": "La pérdida entre val y test es estable y controlada.",
                                    },
                                    {
                                        "mensaje": "Estabilidad cross-project",
                                        "valor": f"LOPO mean F2 = {float(evidence.get('lopo_cal_f2_mean', 0.9546)):.4f}",
                                        "uso": "El desempeño se mantiene al mover el modelo entre proyectos.",
                                    },
                                ],
                                columns=[{"name": c, "id": c} for c in ["mensaje", "valor", "uso"]],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card table-card",
                    ),
                ],
                className="chart-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Lectura rápida para dirección", className="section-title"),
                            dash_table.DataTable(
                                data=[
                                    {
                                        "kpi": "Champion test F2",
                                        "value": f"{float(evidence.get('test_f2', 0.9655)):.4f}",
                                        "interpretation": "El modelo captura casi todos los sprints de alto riesgo.",
                                    },
                                    {
                                        "kpi": "Recall test",
                                        "value": f"{float(evidence.get('test_recall', 0.9953)):.4f}",
                                        "interpretation": "Muy pocos casos de alto riesgo quedan fuera del radar.",
                                    },
                                    {
                                        "kpi": "Precision test",
                                        "value": f"{float(evidence.get('test_precision', 0.8623)):.4f}",
                                        "interpretation": "Las alertas tienen señal suficiente para priorizar revisión.",
                                    },
                                    {
                                        "kpi": "Brier test",
                                        "value": f"{float(evidence.get('test_brier', 0.0666)):.4f}",
                                        "interpretation": "Las probabilidades ya son utilizables para priorización.",
                                    },
                                ],
                                columns=[{"name": c, "id": c} for c in ["kpi", "value", "interpretation"]],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card table-card",
                    ),
                    html.Div(
                        [
                            html.H3("Acciones recomendadas", className="section-title"),
                            html.Ul([html.Li(item) for item in EXECUTIVE_ACTIONS], className="clean-list"),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="chart-grid",
            ),
        ],
        className="section-stack",
    )


def make_model_tab() -> html.Div:
    table = dash_table.DataTable(
        data=MODEL_LEADERBOARD.to_dict("records"),
        columns=[
            {"name": "Modelo", "id": "modelo"},
            {"name": "F2", "id": "f2"},
            {"name": "PR-AUC", "id": "pr_auc"},
            {"name": "Precision", "id": "precision"},
            {"name": "Recall", "id": "recall"},
            {"name": "Brier", "id": "brier"},
        ],
        style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
        style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.H3("Comparación de modelos", className="section-title"), table], className="glass-card table-card"),
                    html.Div(
                        [
                            html.H3("Notas", className="section-title"),
                            html.Ul(
                                [
                                    html.Li("El F2 se maximiza en validation y luego se fija el threshold operativo."),
                                    html.Li("El Brier se reporta aparte para evitar mezclar calibración con umbral."),
                                    html.Li("El champion es XGBoost calibrado con isotónica."),
                                ],
                                className="clean-list",
                            ),
                        ],
                        className="glass-card section-card",
                    ),
                ],
                className="chart-grid",
            ),
        ]
    )


def make_calibration_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(figure=build_calibration_figure(), className="chart"),
                    html.Div(
                        [
                            html.H3("Brier por split", className="section-title"),
                            dash_table.DataTable(
                                data=CALIBRATION_ROWS,
                                columns=[
                                    {"name": "Split", "id": "split"},
                                    {"name": "Raw", "id": "raw"},
                                    {"name": "Isotónica", "id": "isotonic"},
                                    {"name": "Reducción", "id": "reduction"},
                                ],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card table-card",
                    ),
                ],
                className="chart-grid",
            ),
            html.Div(
                [
                    image_card("Reliability diagram - sigmoid", "reliability_sigmoid.png", "Antes de la calibración y con Platt scaling."),
                    image_card("Reliability diagram - isotonic", "reliability_isotonic.png", "Calibrador elegido para producción."),
                ],
                className="image-grid",
            ),
        ]
    )


def make_drift_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(figure=build_drift_figure(), className="chart"),
                    html.Div(
                        [
                            html.H3("Monitoring guards", className="section-title"),
                            dash_table.DataTable(
                                data=[
                                    {"guard": "Flagging rate > 85%", "valor": "74.2%", "acción": "Revisar threshold/calibración"},
                                    {"guard": "Brier test > 0.10", "valor": "0.0666", "acción": "En rango"},
                                    {"guard": "LOPO-CV std F2 > 0.05", "valor": "0.037", "acción": "En rango"},
                                ],
                                columns=[
                                    {"name": "Guard", "id": "guard"},
                                    {"name": "Valor actual", "id": "valor"},
                                    {"name": "Acción", "id": "acción"},
                                ],
                                style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                                style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                            ),
                        ],
                        className="glass-card table-card",
                    ),
                ],
                className="chart-grid",
            ),
            html.Div(
                [
                    image_card("Drift temporal", "temporal_drift.png", "F2 y PSI por año."),
                    image_card("Drift simulado", "simulated_drift.png", "Sensibilidad ante +20% y +50% en cycle time."),
                ],
                className="image-grid",
            ),
        ]
    )


def make_explainability_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    image_card("Global SHAP beeswarm", "shap_beeswarm.png", "Distribución de impacto por feature en el test set."),
                    image_card("Global SHAP bar", "shap_bar.png", "Importancia media absoluta."),
                ],
                className="image-grid",
            ),
            html.Div(
                [
                    image_card("PDP top features", "shap_pdp_top5.png", "Efecto funcional de las top features."),
                    image_card("Caso local - TP", "shap_tp_httpclient.png", "Sprint correctamente flaggeado."),
                ],
                className="image-grid",
            ),
            html.Div(
                [
                    image_card("Caso local - TN", "shap_tn_io.png", "Sprint correctamente descartado."),
                    image_card("Caso local - FN", "shap_fn_math.png", "Caso donde las señales no fueron suficientes."),
                ],
                className="image-grid",
            ),
            html.Div(
                [
                    html.H3("Top drivers", className="section-title"),
                    dash_table.DataTable(
                        data=SHAP_FEATURES.to_dict("records"),
                        columns=[
                            {"name": "Feature", "id": "feature"},
                            {"name": "Importancia", "id": "importance"},
                            {"name": "Razón", "id": "reason"},
                        ],
                        style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
                        style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
                    ),
                ],
                className="glass-card table-card",
            ),
        ]
    )


def build_feature_row(inputs: dict[str, Any]) -> pd.DataFrame:
    total_issues = max(inputs["num_bugs_sprint"] + inputs["num_stories_sprint"] + inputs["num_tasks_sprint"], 1)
    bugs = max(inputs["num_bugs_sprint"], 0)
    stories = max(inputs["num_stories_sprint"], 0)
    tasks = max(inputs["num_tasks_sprint"], 0)
    bug_story_ratio = bugs / max(stories + tasks, 1)
    month = int(inputs["sprint_month"])

    row = {
        "num_bugs_sprint": float(bugs),
        "num_stories_sprint": float(stories),
        "num_tasks_sprint": float(tasks),
        "total_issues_sprint": float(total_issues),
        "log_avg_cycle_time": math.log1p(max(inputs["avg_cycle_time_days"], 0)),
        "log_bug_story_ratio": math.log1p(max(bug_story_ratio, 0)),
        "log_total_issues": math.log1p(total_issues),
        "sprint_year": float(int(inputs["sprint_year"])),
        "sprint_month_sin": math.sin(2 * math.pi * month / 12),
        "sprint_month_cos": math.cos(2 * math.pi * month / 12),
        "deploy_frequency_weekly": float(max(inputs["deploy_frequency_weekly"], 0)),
        "change_failure_rate": float(max(inputs["change_failure_rate"], 0)),
        "bsr_missing": int(bugs == 0 or stories == 0),
        "cycle_missing": int(inputs["cycle_missing"]),
        "dora_missing": int(inputs.get("dora_missing", 1)),
        "bugs_per_issue": bugs / total_issues,
        "log_cycle_x_bsr": math.log1p(max(inputs["avg_cycle_time_days"], 0)) * math.log1p(max(bug_story_ratio, 0)),
    }
    return pd.DataFrame([{name: row.get(name, 0.0) for name in FEATURE_COLS}])


def heuristic_predict_probability(inputs: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    total_issues = max(inputs["num_bugs_sprint"] + inputs["num_stories_sprint"] + inputs["num_tasks_sprint"], 1)
    bugs = max(inputs["num_bugs_sprint"], 0)
    stories = max(inputs["num_stories_sprint"], 0)
    tasks = max(inputs["num_tasks_sprint"], 0)
    cycle_days = max(inputs["avg_cycle_time_days"], 0)
    deploy = max(inputs["deploy_frequency_weekly"], 0)
    cfr = max(inputs["change_failure_rate"], 0)
    cycle_missing = int(inputs["cycle_missing"])
    bug_story_ratio = bugs / max(stories + tasks, 1)
    total_signal = math.log1p(total_issues) - 2.5
    raw = (
        -1.90
        + 1.05 * (math.log1p(cycle_days) - 3.0)
        + 1.25 * (bug_story_ratio - 0.30)
        + 0.55 * total_signal
        + 0.35 * cycle_missing
        + 0.25 * (0.4 * cfr + 0.03 * math.log1p(deploy))
    )
    probability = 1.0 / (1.0 + math.exp(-raw))
    contributions = pd.DataFrame(
        [
            {"feature": "log_avg_cycle_time", "value": round(math.log1p(cycle_days), 3), "contribución": round(1.05 * (math.log1p(cycle_days) - 3.0), 3)},
            {"feature": "bugs_per_issue", "value": round(bugs / total_issues, 3), "contribución": round(1.25 * (bug_story_ratio - 0.30), 3)},
            {"feature": "log_total_issues", "value": round(math.log1p(total_issues), 3), "contribución": round(0.55 * total_signal, 3)},
            {"feature": "cycle_missing", "value": cycle_missing, "contribución": round(0.35 * cycle_missing, 3)},
            {"feature": "change_failure_rate", "value": round(cfr, 3), "contribución": round(0.25 * (0.4 * cfr + 0.03 * math.log1p(deploy)), 3)},
        ]
    ).sort_values("contribución", ascending=False)
    return probability, contributions


def gauge_figure(probability: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability,
            number={"valueformat": ".3f"},
            delta={"reference": OPERATING_THRESHOLD},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#22d3ee"},
                "steps": [
                    {"range": [0, 0.22], "color": "#073b4c"},
                    {"range": [0.22, 0.5], "color": "#134e4a"},
                    {"range": [0.5, 1.0], "color": "#374151"},
                ],
                "threshold": {"line": {"color": "#f97316", "width": 4}, "thickness": 0.75, "value": OPERATING_THRESHOLD},
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_result_card(probability: float, source_label: str) -> html.Div:
    state = "Riesgo alto" if probability >= OPERATING_THRESHOLD else "Riesgo contenido"
    return html.Div(
        [
            html.Div([html.Span("Resultado", className="section-kicker"), pill("flagged" if probability >= OPERATING_THRESHOLD else "clear", "danger" if probability >= OPERATING_THRESHOLD else "ready")], className="result-header"),
            html.Div(state, className="result-title"),
            html.Div(f"Probabilidad: {probability:.3f} · Threshold: {OPERATING_THRESHOLD:.3f}", className="result-copy"),
            html.Div(source_label, className="result-source"),
        ],
        className="glass-card result-card",
    )


def predictor_controls() -> html.Div:
    items = [
        ("Bugs en sprint", "pred-bugs", PREDICTOR_DEFAULTS["num_bugs_sprint"]),
        ("Stories en sprint", "pred-stories", PREDICTOR_DEFAULTS["num_stories_sprint"]),
        ("Tasks en sprint", "pred-tasks", PREDICTOR_DEFAULTS["num_tasks_sprint"]),
        ("Cycle time promedio (días)", "pred-cycle", PREDICTOR_DEFAULTS["avg_cycle_time_days"]),
        ("Año del sprint", "pred-year", PREDICTOR_DEFAULTS["sprint_year"]),
        ("Mes del sprint", "pred-month", PREDICTOR_DEFAULTS["sprint_month"]),
        ("Deploy frequency semanal", "pred-deploy", PREDICTOR_DEFAULTS["deploy_frequency_weekly"]),
        ("Change failure rate", "pred-cfr", PREDICTOR_DEFAULTS["change_failure_rate"]),
    ]
    blocks = [html.Div([html.Label(label), dcc.Input(id=input_id, type="number", value=default, className="number-input")], className="control-block") for label, input_id, default in items]
    blocks.append(
        html.Div(
            [
                html.Label("Flags"),
                dcc.Checklist(id="pred-cycle-missing", options=[{"label": "Sin dato de ciclo", "value": "missing"}], value=[], className="checklist"),
            ],
            className="control-block",
        )
    )
    return html.Div(blocks, className="predictor-controls")


def make_predictor_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Sandbox de predicción", className="section-title"),
                            html.P("El champion del registry se carga si está disponible. Si no responde, se usa un fallback heurístico alineado con los drivers del modelo.", className="body-copy"),
                            predictor_controls(),
                        ],
                        className="glass-card section-card predictor-card",
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="predictor-gauge", className="chart"),
                            html.Div(id="predictor-result"),
                            html.Div(id="predictor-source", className="predictor-source-card"),
                        ],
                        className="glass-card predictor-output-card",
                    ),
                ],
                className="predictor-grid",
            ),
            html.Div([html.H3("Feature row construida", className="section-title"), html.Div(id="predictor-feature-table")], className="glass-card section-card"),
        ],
        className="section-stack",
    )


def make_sources_tab() -> html.Div:
    mode, label, detail, _ = build_runtime_summary()
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Fuentes y trazabilidad", className="section-title"),
                    html.Ul(
                        [
                            html.Li("Bronze: PROMISE, Apache Jira, Red Hat Jira, GHArchive."),
                            html.Li("Silver: outputs limpios y particionados en GCS."),
                            html.Li("Gold: sprint_features en BigQuery."),
                            html.Li("ML: tracking y registry en MLflow."),
                        ],
                        className="clean-list",
                    ),
                    html.Div([html.Span("Estado del modelo", className="section-kicker"), pill(mode, "ready" if mode == "live" else "danger")], className="result-header"),
                    html.Div(label, className="result-title"),
                    html.Div(detail, className="result-copy"),
                    html.A("Abrir MLflow", href=MLFLOW_TRACKING_URI, target="_blank", className="hero-link"),
                ],
                className="glass-card section-card",
            ),
        ]
    )


app = Dash(__name__, title="ShiftMetrics SI7009", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1("ShiftMetrics", className="hero-title"),
                        html.P(
                            "Predicción de defectos escapados con pipeline de datos, entrenamiento y visualización ejecutiva.",
                            className="hero-copy-text",
                        ),
                    ],
                    className="hero-copy glass-card",
                ),
                html.Div(
                    [
                        html.Div([html.Span("Modelo", className="mini-label"), html.Span(f"{SUMMARY['registry']} {SUMMARY['version']}", className="hero-tag")], className="hero-line"),
                        html.Div([html.Span("Threshold", className="mini-label"), html.Span(f"{OPERATING_THRESHOLD:.3f}", className="pill pill-neutral")], className="hero-line"),
                        html.Div([html.Span("MLflow", className="mini-label"), html.A(MLFLOW_TRACKING_URI, href=MLFLOW_TRACKING_URI, target="_blank", className="hero-link")], className="hero-line hero-line-link"),
                    ],
                    className="hero-panel glass-card",
                ),
            ],
            className="hero-shell",
        ),
        dcc.Tabs(
            id="tabs",
            value="executive",
            className="tabs-shell",
            children=[
                dcc.Tab(label="Executive", value="executive", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Modelo", value="model", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Calibración", value="calibration", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Drift", value="drift", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Explainability", value="explainability", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Design Lab", value="design_lab", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Conceptos", value="concepts", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Predictor", value="predictor", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Sources", value="sources", className="tab", selected_className="tab-selected"),
            ],
        ),
        html.Div(id="tab-content", className="content-wrap"),
        make_footer(),
    ],
    className="page-shell",
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab_value: str) -> html.Div:
    if tab_value == "executive":
        return make_executive_tab()
    if tab_value == "model":
        return make_model_tab()
    if tab_value == "calibration":
        return make_calibration_tab()
    if tab_value == "drift":
        return make_drift_tab()
    if tab_value == "explainability":
        return make_explainability_tab()
    if tab_value == "design_lab":
        return make_design_lab_tab()
    if tab_value == "concepts":
        return make_concepts_tab()
    if tab_value == "predictor":
        return make_predictor_tab()
    if tab_value == "sources":
        return make_sources_tab()
    return make_overview()


@app.callback(
    Output("predictor-gauge", "figure"),
    Output("predictor-result", "children"),
    Output("predictor-source", "children"),
    Output("predictor-feature-table", "children"),
    Input("pred-bugs", "value"),
    Input("pred-stories", "value"),
    Input("pred-tasks", "value"),
    Input("pred-cycle", "value"),
    Input("pred-year", "value"),
    Input("pred-month", "value"),
    Input("pred-deploy", "value"),
    Input("pred-cfr", "value"),
    Input("pred-cycle-missing", "value"),
)
def update_predictor(bugs, stories, tasks, cycle_days, year, month, deploy, cfr, cycle_missing_values):
    dora_missing = 1 if (deploy is None or cfr is None) else 0
    inputs = {
        "num_bugs_sprint": int(bugs or 0),
        "num_stories_sprint": int(stories or 0),
        "num_tasks_sprint": int(tasks or 0),
        "avg_cycle_time_days": float(cycle_days or 0),
        "sprint_year": int(year or 2019),
        "sprint_month": int(month or 6),
        "deploy_frequency_weekly": float(deploy or 0),
        "change_failure_rate": float(cfr or 0),
        "cycle_missing": 1 if cycle_missing_values else 0,
        "dora_missing": dora_missing,
    }

    runtime_mode, runtime_label, runtime_detail, model = build_runtime_summary()
    if model is not None:
        try:
            feature_row = build_feature_row(inputs)
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(feature_row)[0][1])
            else:
                probability = float(model.predict(feature_row)[0])
            source_label = f"Fuente: champion live · {runtime_label}"
        except Exception as exc:
            probability, feature_contribs = heuristic_predict_probability(inputs)
            source_label = f"Fallback heurístico · {exc}"
            return gauge_figure(probability, source_label), make_result_card(probability, source_label), runtime_detail, dash_table.DataTable(data=feature_contribs.to_dict("records"), columns=[{"name": c, "id": c} for c in feature_contribs.columns], style_table={"overflowX": "auto"})
    else:
        probability, feature_contribs = heuristic_predict_probability(inputs)
        source_label = f"Fallback heurístico · {runtime_detail}"
        return gauge_figure(probability, source_label), make_result_card(probability, source_label), runtime_detail, dash_table.DataTable(data=feature_contribs.to_dict("records"), columns=[{"name": c, "id": c} for c in feature_contribs.columns], style_table={"overflowX": "auto"})

    _, feature_contribs = heuristic_predict_probability(inputs)
    feature_table = dash_table.DataTable(
        data=feature_contribs.to_dict("records"),
        columns=[{"name": c, "id": c} for c in feature_contribs.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
        style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
    )
    return gauge_figure(probability, source_label), make_result_card(probability, source_label), runtime_label, feature_table


@app.callback(
    Output("viz-story-content", "children"),
    Output("viz-advice-box", "children"),
    Output("viz-variable-box", "children"),
    Input("viz-chart-kind", "value"),
    Input("viz-theme", "value"),
    Input("viz-audience", "value"),
    Input("viz-variable-type", "value"),
)
def update_visual_lab(chart_kind: str, theme: str, audience: str, variable_type: str):
    story_content = build_story_content(chart_kind, theme)
    advice_box = build_story_advice(audience, chart_kind)
    variable_box = build_variable_advice(variable_type)
    return story_content, advice_box, variable_box


def main() -> None:
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()