# ShiftMetrics — Dashboard

Visualizacion del modelo de prediccion de defectos escapados en sprints.
Construido en Streamlit, sin dependencias externas en runtime (todo el dato
viene de snapshots parquet generados offline).

## Como correr en local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Regenerar snapshots desde cero
python generar_snapshot.py

# 3. Levantar el dashboard
streamlit run app.py
```

El dashboard abre en `http://localhost:8501`.

## Estructura

```
shiftmetrics_app/
├── app.py                          # Pagina de inicio
├── pages/
│   ├── 1_🚦_Riesgo_del_periodo.py  # Ranking principal de sprints
│   ├── 2_🔬_Simulador.py            # Predictor interactivo
│   ├── 3_🧠_Cómo_decide.py          # Explicabilidad y confianza
│   └── 4_📡_Salud_del_modelo.py     # Drift y degradacion
├── data/
│   ├── test_snapshot.parquet       # Sprints 2019-2021 con probas
│   ├── shap_global.parquet         # Ranking SHAP global
│   ├── drift_psi.parquet           # PSI por variable
│   ├── drift_yearly.parquet        # F2 por año
│   ├── reliability_curve.parquet   # Curva de calibracion
│   ├── model_metrics.json          # Metricas finales en test
│   └── champion.pkl                # Modelo XGBoost calibrado
├── utils/
│   ├── data.py                     # Loaders cacheados
│   └── estilo.py                   # Paleta y helpers de plotly
└── requirements.txt
```

## Deploy en Streamlit Community Cloud

1. Subir este folder como repo en GitHub.
2. En `share.streamlit.io`, conectar el repo.
3. Main file: `app.py`. Branch: `main`. Python version: 3.11+.
4. El deploy es automatico — no requiere variables de entorno ni secretos.

## Filosofia de diseño

Cada pagina responde una unica pregunta de negocio:

| Pagina | Pregunta |
|--------|----------|
| Inicio | Que hace este dashboard y para que sirve |
| Riesgo del periodo | En que sprints concentrar el esfuerzo de revision |
| Simulador | Como cambia el riesgo al variar las metricas de un sprint |
| Como decide | Por que confiar en el modelo para priorizar |
| Salud del modelo | Esta el modelo deteriorandose, hay que reentrenar |

Reglas de diseño aplicadas:
- Una visualizacion central por pagina con elementos minimos de apoyo
- Color con jerarquia: rojo = atencion, ambar = monitorear, verde = OK,
  azul = identidad corporativa (no para destacar)
- Maximo 4-7 elementos por pantalla, respetando la carga cognitiva
- Lenguaje natural en titulos y narrativas, sin jerga tecnica innecesaria
- Sin tablas duplicadas, sin metricas repetidas entre paginas

## Notas tecnicas

- El champion (XGBoost calibrado con isotonic) se carga desde `data/champion.pkl`.
- El dashboard NO se conecta a MLflow ni BigQuery en runtime para garantizar
  que funciona sin red ni credenciales.
- Los snapshots son reproducibles: corriendo `generar_snapshot.py` se
  reconstruyen todos los parquets y el modelo.
- Para conectar al MLflow real, modificar `utils/data.py` y agregar un
  cargador con `mlflow.sklearn.load_model("models:/...")` como alternativa.
