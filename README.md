# ShiftMetrics — Bronze / Silver / Gold / ML

Pipeline de datos y machine learning para predicción de defectos escapados en sprints de software, usando datos de Apache JIRA, PROMISE, GHArchive y BigQuery sobre GCP.

---

## Estructura del proyecto

```
Shiftmetrics-bronze-silver/
├── Jobs/          # PySpark jobs Bronze → Silver
├── EDAS/          # Análisis exploratorios por fuente
├── Gold/          # Tablas BigQuery (sprint_features)
├── SI7009/        # Pipeline de ML (entrenamiento, evaluación, modelos)
├── mlflow-server/ # Despliegue del servidor MLflow en Cloud Run
└── docs/          # Documentación general
```

---

## Bronze / Silver — Jobs PySpark

> Documentación pendiente.

---

## EDAS — Análisis Exploratorio

> Documentación pendiente.

---

## Gold — Feature Store en BigQuery

> Documentación pendiente.

---

## ML — Predicción de Defectos Escapados (SI7009)

Predice si un sprint producirá un **defecto escapado a producción** (`defecto_escapado = 1`), usando datos de Jira en BigQuery. El pipeline corre enteramente en **Vertex AI Workbench** sobre GCP; los experimentos se registran en MLflow (Cloud Run).

### Infraestructura

| Componente | Detalle |
|---|---|
| **Entorno de ejecución** | Vertex AI Workbench `shiftmetrics-ml` — n1-standard-8, us-central1-a |
| **Feature store** | BigQuery `shiftmetrics-analytics.shiftmetrics_gold.sprint_features` |
| **MLflow tracking** | Cloud Run — `https://mlflow-server-919593201130.us-central1.run.app` |
| **Artifact store** | `gs://shiftmetrics-bronze/mlruns/` |
| **HPO storage** | SQLite persistente en el Workbench (`optuna_shiftmetrics.db`) |

### Qué es MLflow

MLflow es la plataforma que registra todo lo que hace el pipeline: parámetros, métricas, modelos y artefactos de cada experimento. El servidor corre en Cloud Run y tiene una UI accesible desde el navegador en:

```
https://mlflow-server-919593201130.us-central1.run.app
```

Desde ahí se pueden comparar runs, ver métricas por experimento y descargar modelos sin necesidad de estar en el Workbench.

---

### Acceder al Workbench

Abrir JupyterLab directamente en el navegador, autenticado con la cuenta que tiene acceso al proyecto `shiftmetrics-analytics`:

```
https://6a58f6be0973f770-dot-us-central1.notebooks.googleusercontent.com
```

**Si la instancia está detenida**, iniciarla desde la terminal de la laptop:

```bash
gcloud workbench instances start shiftmetrics-ml \
  --project=shiftmetrics-analytics \
  --location=us-central1-a
```

Detenerla cuando no se use para evitar costos:

```bash
gcloud workbench instances stop shiftmetrics-ml \
  --project=shiftmetrics-analytics \
  --location=us-central1-a
```

Verificar el estado:

```bash
gcloud workbench instances describe shiftmetrics-ml \
  --project=shiftmetrics-analytics \
  --location=us-central1-a \
  --format="value(state)"
# Resultado esperado: ACTIVE
```

---

### Qué hacer una vez dentro de JupyterLab

#### Paso 1 — Abrir una terminal

En JupyterLab verás una pantalla con íconos. Busca el ícono que dice **Terminal** y haz clic en él.

Si no lo ves directamente, ve al menú superior: **File → New → Terminal**.

Se abrirá una terminal negra donde puedes escribir comandos.

#### Paso 2 — Navegar al código

El código ya está en el Workbench. Ir a esa carpeta:

```bash
cd /home/samargo1703_gmail_com/shiftmetrics/SI7009
```

Confirmar con `ls`:

```
baselines.py  calibration.py  config.py  drift.py  feature_store.py
metrics.py    run_pipeline.py  shap_analysis.py  threshold.py  train_gbm.py  train_lr.py
```

#### Paso 3 — Actualizar el código

```bash
git pull origin main
```

#### Paso 4 — Instalar dependencias (solo la primera vez)

```bash
pip install -r requirements.txt
```

> Las credenciales de GCP ya están configuradas en el Workbench — no necesitas hacer login.

#### Verificar que todo está listo

```bash
# Confirmar acceso a BigQuery
python -c "
from google.cloud import bigquery
client = bigquery.Client(project='shiftmetrics-analytics')
result = list(client.query('SELECT COUNT(*) as n FROM \`shiftmetrics-analytics.shiftmetrics_gold.sprint_features\`').result())
print('Filas en BigQuery:', result[0].n)
"
# Resultado esperado: Filas en BigQuery: 42747

# Confirmar que MLflow responde
python -c "import mlflow; mlflow.set_tracking_uri('https://mlflow-server-919593201130.us-central1.run.app'); print(mlflow.get_tracking_uri())"
```

---

### Ejecutar el pipeline completo

```bash
cd /home/samargo1703_gmail_com/shiftmetrics/SI7009

python run_pipeline.py --n-trials 50
```

Duración estimada en el Workbench (n1-standard-8): **~45 minutos**.

> `--n-trials 50` es el valor estándar definido en `config.py`. Si el archivo `optuna_shiftmetrics.db` ya existe, Optuna retoma los trials completados automáticamente.

#### Las 10 rondas del pipeline

```
[0]  Feature matrix      — Carga BigQuery + feature engineering + split temporal 4-way
[1]  Baselines           — MajorityClass y ThresholdBSR
[2]  Logistic Regression — Grid 18 configs + LOPO-CV
[3]  XGBoost             — Optuna TPE (50 trials, F2, SQLite persistente)
[4]  LightGBM            — Optuna TPE (50 trials, challenger)
[5]  Champion/Challenger — Selección por ΔF2 > 0.005 + McNemar test
[6]  Calibración         — Platt scaling vs. isotonic (holdout 2015)
[7]  LOPO-CV GBM         — Leave-One-Project-Out con hiperparámetros congelados
[8]  SHAP                — Importancia global, local, PDPs
[9]  Drift               — KS + PSI temporal y simulado (+20%/+50%)
[10] Threshold           — F2-óptimo y ROI-óptimo; evaluación final en test
```

#### Opciones disponibles

```bash
python run_pipeline.py --help

# Prueba rápida (~5 min)
python run_pipeline.py --n-trials 5

# Saltar Logistic Regression (ronda 2)
python run_pipeline.py --n-trials 50 --skip-lr

# Saltar SHAP y drift
python run_pipeline.py --n-trials 50 --skip-shap --skip-drift

# Guardar el modelo en ruta personalizada
python run_pipeline.py --n-trials 50 --champion-out ~/mis_modelos/champion_v5.pkl
```

#### Reanudar si el pipeline se interrumpe

Optuna guarda el progreso en `optuna_shiftmetrics.db`. Si la sesión cae, volver a correr el mismo comando:

```bash
python run_pipeline.py --n-trials 50
# Verás: "Reanudando estudio existente (N trials completados)"
```

Para empezar desde cero:

```bash
rm optuna_shiftmetrics.db
python run_pipeline.py --n-trials 50
```

---

### Ejecutar módulos individuales

Cada archivo puede correrse por separado. Útil para depurar una etapa sin correr todo el pipeline:

```bash
python feature_store.py    # Verificar carga de datos y feature engineering
python baselines.py        # Solo baselines (también verifica conexión a BigQuery)
python train_gbm.py        # Solo XGBoost + LightGBM con Optuna
python calibration.py      # Solo calibración
python threshold.py        # Solo selección de threshold
python drift.py            # Solo análisis de drift
python shap_analysis.py    # Solo análisis SHAP
```

---

### Consultar MLflow

#### Interfaz web

Abrir desde el navegador (no requiere estar en el Workbench):

```
https://mlflow-server-919593201130.us-central1.run.app
```

Desde la UI se puede:
- Ver todos los runs del experimento **`shiftmetrics-sprint-defect`**
- Comparar modelos por F2, Brier, PR-AUC
- Descargar artefactos (modelos `.pkl`, plots, CSVs de drift y threshold)
- Gestionar versiones en **Models → ShiftMetrics-DefectoEscapado**

#### Consultar el Model Registry por código

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="https://mlflow-server-919593201130.us-central1.run.app")

# Ver todas las versiones registradas
for v in client.search_model_versions("name='ShiftMetrics-DefectoEscapado'"):
    print(f"v{v.version} | aliases: {v.aliases} | run_id: {v.run_id}")

# Versión en producción
version_prod = client.get_model_version_by_alias("ShiftMetrics-DefectoEscapado", "production")
print(f"Producción → v{version_prod.version}")

# Versión champion actual
version_champ = client.get_model_version_by_alias("ShiftMetrics-DefectoEscapado", "champion")
print(f"Champion → v{version_champ.version}")
```

#### Cargar el modelo directamente desde el registry

```python
import mlflow.sklearn

mlflow.set_tracking_uri("https://mlflow-server-919593201130.us-central1.run.app")

# Por alias (recomendado)
model = mlflow.sklearn.load_model("models:/ShiftMetrics-DefectoEscapado@production")

# Por versión específica
model_v4 = mlflow.sklearn.load_model("models:/ShiftMetrics-DefectoEscapado/4")
```

---

### Cargar y usar el modelo calibrado

#### Desde el archivo local generado por el pipeline

```python
import joblib
import numpy as np

model = joblib.load("/tmp/champion_calibrated.pkl")

# Features en el mismo orden que FEATURE_COLS en SI7009/config.py:
# num_bugs_sprint, num_stories_sprint, num_tasks_sprint, total_issues_sprint,
# log_avg_cycle_time, log_bug_story_ratio, log_total_issues,
# sprint_year, sprint_month_sin, sprint_month_cos,
# deploy_frequency_weekly, change_failure_rate,
# bsr_missing, cycle_missing, dora_missing,
# bugs_per_issue, log_cycle_x_bsr

X = np.array([[5, 8, 3, 16, 3.2, 1.1, 2.8, 2020, 0.5, 0.87, 0, 0, 0, 0, 1, 0.31, 3.52]])

prob = model.predict_proba(X)[:, 1][0]
print(f"Probabilidad de defecto escapado: {prob:.3f}")

THRESHOLD = 0.220  # threshold operacional (F2-óptimo y ROI-óptimo)
print(f"Predicción: {'RIESGO ALTO' if prob >= THRESHOLD else 'riesgo bajo'}")
```

#### Sobre datos reales desde BigQuery

```python
from feature_store import get_feature_matrix
from config import FEATURE_COLS
import joblib

train, cal, val, test, df_full = get_feature_matrix()
model = joblib.load("/tmp/champion_calibrated.pkl")

X_test = test[FEATURE_COLS].values
prob_test = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.220
pred_test = (prob_test >= THRESHOLD).astype(int)

print(f"Sprints evaluados : {len(pred_test)}")
print(f"Sprints en riesgo : {pred_test.sum()} ({pred_test.mean():.1%})")
```

#### Threshold operacional

| Criterio | Valor | Cuándo usar |
|---|---|---|
| **F2-óptimo** | **0.220** | Producción — maximiza recall (FN penalizado 2×) |
| **ROI-óptimo** | **0.220** | Ambos convergen; FN cuesta 3× un FP |

> Si el flagging rate supera el **85%**, recalibrar o subir el threshold.

---

### Archivos de salida

El pipeline deja los siguientes archivos en `/tmp/` del Workbench y los sube a MLflow:

| Archivo | Descripción |
|---|---|
| `champion_calibrated.pkl` | Modelo champion calibrado |
| `leaderboard.csv` | Tabla comparativa de todos los modelos |
| `pipeline_summary.json` | Resumen JSON con champion, thresholds, métricas y CIs |
| `reliability_sigmoid.png` | Reliability diagram — Platt scaling |
| `reliability_isotonic.png` | Reliability diagram — Isotonic regression |
| `pr_curve.png` | Curva Precision-Recall con marcadores de threshold |
| `threshold_tradeoff.png` | F2, precision, recall y business value vs threshold |
| `threshold_sweep.csv` | Métricas por threshold en [0.05, 0.95] |
| `temporal_drift.png` | F2 y PSI por año (2000-2021) |
| `simulated_drift.png` | Degradación bajo shifts +20%/+50% en cycle_time |
| `ks_drift_report.csv` | KS statistic y PSI por feature (train vs test) |
| `optuna_shiftmetrics.db` | Historial de trials de Optuna (resumable) |

---

### Troubleshooting

**La instancia del Workbench no responde o está detenida**
```bash
gcloud workbench instances start shiftmetrics-ml \
  --project=shiftmetrics-analytics \
  --location=us-central1-a
```

**`ModuleNotFoundError` al correr el pipeline**
```bash
pip install -r requirements.txt
```

**MLflow no responde al iniciar el pipeline**
```bash
curl -s https://mlflow-server-919593201130.us-central1.run.app/health
# Esperado: {"status": "OK"}
```
Si no responde, esperar ~30 segundos y reintentar.

**Optuna empieza desde cero en vez de reanudar**

Verificar que `optuna_shiftmetrics.db` está en `/home/samargo1703_gmail_com/shiftmetrics/SI7009/`. Si se corrió el pipeline desde otro directorio, el `.db` quedó en otro lugar.

**El modelo no se registra en el MLflow Model Registry**
```python
import mlflow
mlflow.set_tracking_uri("https://mlflow-server-919593201130.us-central1.run.app")
# Reemplazar <RUN_ID> con el run_id del run de calibración (se imprime en consola)
mv = mlflow.register_model("runs:/<RUN_ID>/calibrated_model", "ShiftMetrics-DefectoEscapado")
print(f"Versión registrada: {mv.version}")
```

**Error de memoria durante SHAP**
```bash
python run_pipeline.py --n-trials 50 --skip-shap
```

---

### Referencias

- **Informe técnico**: `SI7009/docs/SI7009_informe_tecnico.md`
- **Model Card**: `SI7009/model_card.md`
- **Configuración centralizada**: `SI7009/config.py`
- **MLflow UI**: https://mlflow-server-919593201130.us-central1.run.app
- **Consola GCP**: https://console.cloud.google.com → proyecto `shiftmetrics-analytics`
