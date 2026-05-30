# ShiftMetrics — Dash / ML

Dashboard ejecutivo en Dash para la predicción de defectos escapados en sprints,
acompañado por el pipeline de machine learning y la configuración del modelo.

## Cómo correr en local

```bash
pip install -r requirements.txt
python app.py
```

La app abre en `http://localhost:8050`.

## Estructura

```
shiftmetrics-bronze-silver/
├── app.py              # Launcher de Dash
├── config.py           # Reexporta la configuración central
├── ml/                 # Pipeline, calibración, drift, SHAP y dashboard Dash
├── docs/               # Documentación del proyecto
├── infra/              # Infraestructura como código
├── README.md
└── requirements.txt
```

## Dashboard

El dashboard Dash vive en `ml/dashboard_app.py`. Incluye:

- Vista ejecutiva
- Comparación de modelos
- Calibración
- Drift
- Explainability
- Predictor interactivo

## Notas

- El dashboard se ejecuta sin Streamlit.
- `app.py` solo arranca la aplicación Dash.
- `config.py` es un wrapper para mantener compatibles los imports del paquete `ml`.
