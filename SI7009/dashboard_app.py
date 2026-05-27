"""ShiftMetrics SI7009 dashboard.

Dash app for the visualization layer of the project.
The dashboard unifies pipeline status, model comparison, calibration,
drift, SHAP evidence, and an interactive prediction sandbox.
"""

from __future__ import annotations

import base64
import math
from datetime import date
from pathlib import Path
from typing import Any

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
        {"layer": "Viz", "status": "In progress", "detail": "Dash UI consolidating all evidence"},
    ]
)

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

CHAMPION_URI_CANDIDATES = [
    f"models:/{MODEL_REGISTRY_NAME}@champion",
    f"models:/{MODEL_REGISTRY_NAME}@production",
    f"models:/{MODEL_REGISTRY_NAME}/{MODEL_VERSION.lstrip('v')}",
]


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
    bug_story_ratio = bugs / max(stories, 1)
    month = int(inputs["sprint_month"])

    row = {
        "num_bugs_sprint": float(bugs),
        "num_stories_sprint": float(stories),
        "num_tasks_sprint": float(max(inputs["num_tasks_sprint"], 0)),
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
        "dora_missing": 1,
        "bugs_per_issue": bugs / total_issues,
        "log_cycle_x_bsr": math.log1p(max(inputs["avg_cycle_time_days"], 0)) * math.log1p(max(bug_story_ratio, 0)),
    }
    return pd.DataFrame([{name: row.get(name, 0.0) for name in FEATURE_COLS}])


def heuristic_predict_probability(inputs: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    total_issues = max(inputs["num_bugs_sprint"] + inputs["num_stories_sprint"] + inputs["num_tasks_sprint"], 1)
    bugs = max(inputs["num_bugs_sprint"], 0)
    stories = max(inputs["num_stories_sprint"], 0)
    cycle_days = max(inputs["avg_cycle_time_days"], 0)
    deploy = max(inputs["deploy_frequency_weekly"], 0)
    cfr = max(inputs["change_failure_rate"], 0)
    cycle_missing = int(inputs["cycle_missing"])
    bug_story_ratio = bugs / max(stories, 1)
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
                        html.Div("ShiftMetrics", className="eyebrow"),
                        html.H1("Centro de visualización", className="hero-title"),
                        html.P(
                            "Una sola interfaz para entender el pipeline, comparar modelos, revisar calibración y drift, y probar predicciones con un champion opcional desde MLflow.",
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
            value="overview",
            className="tabs-shell",
            children=[
                dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Modelo", value="model", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Calibración", value="calibration", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Drift", value="drift", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Explainability", value="explainability", className="tab", selected_className="tab-selected"),
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
    if tab_value == "model":
        return make_model_tab()
    if tab_value == "calibration":
        return make_calibration_tab()
    if tab_value == "drift":
        return make_drift_tab()
    if tab_value == "explainability":
        return make_explainability_tab()
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
            return make_result_card(probability, source_label), gauge_figure(probability, source_label), runtime_detail, dash_table.DataTable(data=feature_contribs.to_dict("records"), columns=[{"name": c, "id": c} for c in feature_contribs.columns], style_table={"overflowX": "auto"})
    else:
        probability, feature_contribs = heuristic_predict_probability(inputs)
        source_label = f"Fallback heurístico · {runtime_detail}"
        return make_result_card(probability, source_label), gauge_figure(probability, source_label), runtime_detail, dash_table.DataTable(data=feature_contribs.to_dict("records"), columns=[{"name": c, "id": c} for c in feature_contribs.columns], style_table={"overflowX": "auto"})

    _, feature_contribs = heuristic_predict_probability(inputs)
    feature_table = dash_table.DataTable(
        data=feature_contribs.to_dict("records"),
        columns=[{"name": c, "id": c} for c in feature_contribs.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#0f172a", "color": "#e2e8f0", "fontWeight": "700"},
        style_cell={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontFamily": "Aptos, Segoe UI, sans-serif", "padding": "10px"},
    )
    return make_result_card(probability, source_label), gauge_figure(probability, source_label), runtime_label, feature_table


def main() -> None:
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()