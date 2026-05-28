# ShiftMetrics — Bronze / Silver / Gold / ML

Pipeline de datos y machine learning para predicción de defectos escapados en sprints de software, usando datos de Apache JIRA, PROMISE, GHArchive y BigQuery sobre GCP.

**Proyecto:** Predicción de Defect Escape en Sprints de Software
**Curso:** SI7006 · Proyecto Integrador · Master Data Science & Analytics · EAFIT
**Equipo:** saldsfishy@gmail.com · samargo1703@gmail.com · freddycadavid2015@gmail.com · juanprc2017@gmail.com · juanmejia0317@gmail.com
**Versión:** 1.0 · Mayo 2026

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

## 1. Descripción General

ShiftMetrics Analytics es un sistema de Machine Learning para predecir el defect escape en sprints de desarrollo de software. El defect escape es la proporción de defectos que no se detectan durante el sprint y llegan a producción.

El proyecto usa datos históricos de repositorios de código, Jira (Apache y Red Hat), logs de eventos de GitHub y métricas de calidad de proyectos open source para entrenar modelos que permitan identificar riesgos antes del cierre del sprint.

**Estado al cierre (Mayo 2026):**
- Infraestructura GCP configurada (proyecto, IAM, buckets, BigQuery, APIs)
- Datasets crudos cargados en la capa Bronze (4.36 GiB total)
- Conversión BSON → Parquet del dataset Apache Jira (6 colecciones, ~13M filas)
- EDA ejecutado sobre los 4 datasets
- 4 jobs Silver ejecutados en Dataproc
- Job Gold ejecutado: tabla `sprint_features` con 42,747 filas en BigQuery
- ML Pipeline ejecutado con XGBoost + LightGBM + calibración + SHAP
- Dashboard Dash y servidor MLflow desplegados en Cloud Run

---

## 2. Arquitectura del Pipeline (Medallion)

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│   PROMISE   │   │ Apache Jira │   │ Red Hat Jira │   │  GHArchive  │
│  (GitHub)   │   │  (Zenodo)   │   │   (Zenodo)   │   │   (2022)    │
└──────┬──────┘   └──────┬──────┘   └──────┬───────┘   └──────┬──────┘
       │                 │                  │                   │
       └─────────────────┴──────────────────┴───────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    shiftmetrics-bronze (GCS)   │
                   │         4.36 GiB · 176+ obj    │
                   └───────────────┬───────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │  Dataproc PySpark    │
                        │  (BSON→Parquet +     │
                        │   4 Silver Jobs)     │
                        └──────────┬──────────┘
                                   │
                   ┌───────────────────────────────┐
                   │    shiftmetrics-silver (GCS)   │
                   │      4 tablas Parquet          │
                   └───────────────┬───────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │  Dataproc Gold Job   │
                        │  PySpark + BQ        │
                        └──────────┬──────────┘
                                   │
                   ┌───────────────────────────────┐
                   │   BigQuery: shiftmetrics_gold  │
                   │   sprint_features: 42,747 f.   │
                   └───────────────┬───────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │   ML Pipeline        │
                        │   Vertex AI          │
                        └──────────┬──────────┘
                                   │
                        ┌──────────┴──────────┐
                        │   Deploy API REST    │
                        │   Cloud Run          │
                        └─────────────────────┘
```

| Capa   | Bucket GCS               | Descripción                               |
|--------|--------------------------|-------------------------------------------|
| Bronze | `shiftmetrics-bronze`    | Datos crudos sin transformar              |
| Silver | `shiftmetrics-silver`    | Datos limpios, estandarizados, Parquet    |
| Gold   | `shiftmetrics_gold` (BQ) | Features y tablas analíticas para ML      |

---

## 3. Infraestructura GCP

| Parámetro        | Valor                                         |
|------------------|-----------------------------------------------|
| Project Name     | shiftmetrics-analytics                        |
| Project Number   | 919593201130                                  |
| Region           | us-central1                                   |
| Cuota CPUs       | 12 vCPUs máximos (cuenta de prueba)           |

**Buckets (privados, us-central1):**

| Bucket                  | Estado                                   |
|-------------------------|------------------------------------------|
| `shiftmetrics-bronze`   | Activo — 4.36 GiB, 176+ objetos         |
| `shiftmetrics-silver`   | Activo — 4 tablas Parquet particionadas  |
| `shiftmetrics-gold`     | Creado (datos en BigQuery)               |

**APIs habilitadas:** Cloud Storage, BigQuery, Dataproc, Cloud Run, Vertex AI.

---

## 4. Bronze / Silver — Jobs PySpark

### 4.1 Datasets crudos

| Dataset        | Ruta GCS                  | Archivos   | Tamaño   | Formato  |
|----------------|---------------------------|------------|----------|----------|
| PROMISE        | `bronze/promise/`         | 144 CSV    | 534 MiB  | CSV      |
| Apache Jira    | `bronze/apache-jira/`     | 6 BSON.gz  | 1.9 GiB  | BSON.gz  |
| Red Hat Jira   | `bronze/redhat-jira/`     | 2 ZIP      | 266 MB   | ZIP/CSV  |
| GHArchive 2022 | `bronze/gharchive/`       | 24 JSON.gz | 1.7 GiB  | JSON.gz  |

**Estructura en GCS:**

```
gs://shiftmetrics-bronze/
├── apache-jira/
│   ├── apache-jira-comments.bson.gz   (765 MB)
│   ├── apache-jira-events.bson.gz     (339 MB)
│   ├── apache-jira-issues.bson.gz     (729 MB)
│   ├── apache-jira-projects.bson.gz   (1.2 MB)
│   ├── apache-jira-users.bson.gz      (7.8 MB)
│   └── apache-jira-worklogs.bson.gz   (120 MB)
├── apache-jira-parquet/
│   ├── comments/   [~4.6M filas]
│   ├── events/     [~7.5M filas]
│   ├── issues/     [~978K filas]
│   ├── projects/
│   ├── users/
│   └── worklogs/   [~120K filas]
├── gharchive/          [24 archivos .json.gz, 2022]
├── promise/
│   └── PROMISE-backup/bug-data/  [144 CSV]
├── redhat-jira/
│   ├── redhat-inputs.zip   (usar este)
│   └── redhat-outputs.zip  (NO usar — no son datos Jira)
└── scripts/
    ├── bson_fix.py
    ├── silver_job_01_promise.py
    ├── silver_job_02_apache_jira.py
    ├── silver_job_03_redhat_jira.py
    ├── silver_job_04_gharchive.py
    └── gold_job_sprint_features.py
```

> `redhat-outputs.zip` contiene columnas `Time, beta, alpha, epsilon, gamma` — son datos de modelo matemático, no issues de Jira. No usar.

### 4.2 Conversión BSON a Parquet (Apache Jira)

Apache Jira llegó como dump MongoDB. Spark no puede leer BSON directamente; se convirtió a Parquet con un job Dataproc previo.

| Componente | Valor                                          |
|------------|------------------------------------------------|
| Cluster    | `jira-converter` — single-node                 |
| Master     | n1-highmem-8 (8 vCPUs, 52 GB RAM)             |
| Script     | `gs://shiftmetrics-bronze/scripts/bson_fix.py` |

Single-node porque la cuota máxima es 12 vCPUs y un cluster multi-nodo estándar requería 16.

**Resultado:**

| Colección | BSON.gz | Filas Parquet |
|-----------|---------|---------------|
| projects  | 1.2 MB  | pequeño       |
| users     | 7.8 MB  | pequeño       |
| worklogs  | 120 MB  | ~120K         |
| events    | 339 MB  | ~7.5M         |
| issues    | 729 MB  | ~978K         |
| comments  | 765 MB  | ~4.6M         |
| **TOTAL** | **1.9 GiB** | **~13M filas** |

```bash
# Crear cluster
gcloud dataproc clusters create jira-converter \
  --region=us-central1 --zone=us-central1-a \
  --master-machine-type=n1-highmem-8 \
  --master-boot-disk-size=100 \
  --single-node --image-version=2.1-debian11 \
  --project=shiftmetrics-analytics

# Ejecutar job
gcloud dataproc jobs submit pyspark \
  gs://shiftmetrics-bronze/scripts/bson_fix.py \
  --cluster=jira-converter --region=us-central1 \
  --project=shiftmetrics-analytics
```

### 4.3 Silver Job 01 — PROMISE

```
Input:  gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*/*.csv
Output: gs://shiftmetrics-silver/promise/
Script: silver_job_01_promise.py
```

Schema: `project, version, wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc, bug_count, defect_flag (0/1), defect_density, total_modules` — partición: `project`

Transformaciones: nombres a lowercase, CK a float64, binarizar `bug > 0` → `defect_flag`, extraer `project` y `version` del nombre del archivo.

Cobertura: 11 proyectos (ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces).

### 4.4 Silver Job 02 — Apache Jira

```
Input:  gs://shiftmetrics-bronze/apache-jira-parquet/issues/
Output: gs://shiftmetrics-silver/apache-jira/
Script: silver_job_02_apache_jira.py
```

Schema: `key, project, issuetype_name, issue_category (bug/story/task/other), status, resolution, created_ts, resolution_ts, cycle_time_days, sprint (YYYY-M), bug_story_ratio` — partición: `project`

La columna de tipo es `issuetype_name` (no `issuetype`) porque el BSON fue aplanado por `bson_fix.py`. El proyecto se extrae de `key.split("-")[0]`.

### 4.5 Silver Job 03 — Red Hat Jira

```
Input:  gs://shiftmetrics-bronze/redhat-jira-parquet/  (251 Parquets pre-generados)
Output: gs://shiftmetrics-silver/redhat-jira/
Script: silver_job_03_redhat_jira.py
```

Schema: `issue_key, issue_type, issue_category, status, project_key, project_name, resolution, created_ts, resolved_ts, cycle_time_days, sprint (YYYY-M), num_bugs, num_stories, num_tasks, total_issues_sprint, bug_story_ratio` — partición: `project_key`

Los 251 parquets se generaron localmente desde `redhat-inputs.zip` y se subieron a `gs://shiftmetrics-bronze/redhat-jira-parquet/`. Las fechas están en formato `DD/MM/YYYY HH:MM` — `dayfirst=True` es obligatorio.

### 4.6 Silver Job 04 — GHArchive

```
Input:  gs://shiftmetrics-bronze/gharchive/*.json.gz (24 archivos)
Output: gs://shiftmetrics-silver/gharchive/
Script: silver_job_04_gharchive.py
```

Schema: `repo_name, apache_project_key, year, month, push_count, deploy_frequency_weekly, total_prs_closed, prs_merged, prs_not_merged, change_failure_rate` — partición: `year, month`

El filtro `repo_name.startswith('apache/')` es obligatorio. Sin él, el CFR calculado es 77.3% (todos los repos del mundo); con el filtro es 26.1% (solo proyectos Apache).

---

## 5. EDAS — Análisis Exploratorio

| Notebook           | Dataset                | Resultado principal                        |
|--------------------|------------------------|--------------------------------------------|
| EDA_01_PROMISE     | PROMISE (144 CSV)      | Schema incorrecto detectado; ruta corregida a `bug-data/` |
| EDA_02_APACHE_JIRA | Apache Jira (Parquet)  | Clasificador corregido; 94 columnas en `issues` |
| EDA_03_REDHAT_JIRA | Red Hat Jira (ZIP/CSV) | 505,096 issues, 251 proyectos, `dayfirst=True` |
| EDA_04_GHARCHIVE   | GHArchive 2022         | CFR real: 26.1%; 4,994 eventos Apache      |

**EDA_03 — Red Hat Jira (resultados):**

| Métrica                    | Valor                                 |
|----------------------------|---------------------------------------|
| Total issues               | 505,096                               |
| Proyectos únicos           | 251                                   |
| Issues con resolución      | 436,475 / 505,096 (86.4%)            |
| Cycle Time p50             | 28 días                               |
| Cycle Time p75             | 127 días                              |
| Bug ratio                  | 1.21 (223K bugs / 184K tasks+stories) |

**EDA_04 — GHArchive (resultados):**

| Métrica                    | Valor                                  |
|----------------------------|----------------------------------------|
| Archivos procesados        | 24 de 24 (1.69 GiB)                   |
| Eventos Apache             | 4,994                                  |
| Change Failure Rate real   | 26.1%                                  |
| Rango temporal             | 2022-01-01 → 2022-03-15               |

---

## 6. Gold — Feature Store en BigQuery

**Tabla:** `shiftmetrics-analytics:shiftmetrics_gold.sprint_features`

### JOIN entre capas Silver

```
Silver Apache Jira  (project, sprint)          <- BASE
    LEFT JOIN Silver PROMISE    ON project
    LEFT JOIN Silver GHArchive  ON apache_project_key=project AND year/month=sprint
    LEFT JOIN Silver Red Hat    ON project_key=project AND sprint=sprint
->  BigQuery: shiftmetrics_gold.sprint_features
```

| Fuente Silver  | Filas aportadas |
|----------------|-----------------|
| Apache Jira    | 42,747 (base)   |
| Red Hat Jira   | 17,475          |
| GHArchive      | 205             |
| PROMISE CK     | 11 proyectos    |

### Schema: sprint_features

| Columna                   | Tipo    | Fuente     | Nullable | Descripción                                      |
|---------------------------|---------|------------|----------|--------------------------------------------------|
| `sprint_id`               | STRING  | Generado   | No       | `"{project}_{sprint}"` ej: `"SPARK_2022-3"`     |
| `project`                 | STRING  | Apache Jira| No       | Clave del proyecto                               |
| `sprint`                  | STRING  | Apache Jira| No       | `"YYYY-M"`                                       |
| `avg_wmc`                 | FLOAT64 | PROMISE    | Si       | Weighted Methods per Class                       |
| `avg_dit`                 | FLOAT64 | PROMISE    | Si       | Depth of Inheritance Tree                        |
| `avg_cbo`                 | FLOAT64 | PROMISE    | Si       | Coupling Between Objects                         |
| `avg_rfc`                 | FLOAT64 | PROMISE    | Si       | Response for a Class                             |
| `avg_lcom`                | FLOAT64 | PROMISE    | Si       | Lack of Cohesion of Methods                      |
| `avg_loc`                 | FLOAT64 | PROMISE    | Si       | Lines of Code                                    |
| `defect_density`          | FLOAT64 | PROMISE    | Si       | Modulos defectuosos / total modulos              |
| `bug_story_ratio`         | FLOAT64 | Jira       | No       | Bugs / (Stories + Tasks) por sprint              |
| `avg_cycle_time_days`     | FLOAT64 | Jira       | No       | Tiempo promedio de resolucion (dias)             |
| `num_bugs_sprint`         | INT64   | Jira       | No       | Bugs en el sprint                                |
| `num_stories_sprint`      | INT64   | Jira       | No       | Stories en el sprint                             |
| `deploy_frequency_weekly` | FLOAT64 | GHArchive  | Si       | Pushes a main/master por semana                  |
| `change_failure_rate`     | FLOAT64 | GHArchive  | Si       | PRs sin merge / total PRs cerrados               |
| `defecto_escapado`        | INT64   | Derivado   | No       | TARGET ML: 0 o 1 (avg_cycle_time_days > 30 dias)|

### Verificacion

```sql
SELECT
  COUNT(*)                               AS total_filas,
  COUNT(DISTINCT project)               AS proyectos,
  COUNT(DISTINCT sprint)                AS sprints,
  SUM(defecto_escapado)                 AS defectos_positivos,
  ROUND(AVG(defecto_escapado) * 100, 1) AS prevalencia_pct
FROM shiftmetrics_gold.sprint_features;
```

| total_filas | proyectos | sprints | defectos_positivos | prevalencia_pct |
|-------------|-----------|---------|--------------------|-----------------| 
| 42,747      | 619       | 250     | 30,116             | 70.5%           |

### Notas para ML

**Prevalencia 70.5%:** El target tiene desbalance moderado. Opciones: `scale_pos_weight = 0.42` en XGBoost, undersampling, o ajustar el threshold de cycle_time_days de 30 a 60 dias. El F2-Score ya penaliza los falsos negativos.

**NULLs en metricas CK:** Las columnas `avg_wmc`, `avg_cbo`, `avg_rfc`, `avg_lcom`, `avg_loc` son NULL para 608 de los 619 proyectos. Solo los 11 proyectos PROMISE tienen datos CK. Se recomienda agregar columna binaria `has_ck_data` e imputar por mediana.

**Split temporal:** Usar corte por `sprint` para evitar data leakage.

```python
train = df[df['sprint'] < '2022-8']
val   = df[(df['sprint'] >= '2022-8') & (df['sprint'] < '2022-11')]
test  = df[df['sprint'] >= '2022-11']
```

---

## ML — Predicción de Defectos Escapados (SI7009)

Predice si un sprint producirá un **defecto escapado a producción** (`defecto_escapado = 1`), usando datos de Jira en BigQuery. El pipeline corre enteramente en **Vertex AI Workbench** sobre GCP; los experimentos se registran en MLflow (Cloud Run).

La capa de visualización técnica ahora incluye un dashboard en Dash dentro de `SI7009/dashboard_app.py`, con una vista ejecutiva inicial y pestañas para overview, modelo, calibración, drift, explainability, predictor y fuentes.

Para abrirlo localmente en el repo:

```bash
python SI7009/dashboard_app.py
```

El panel fue diseñado para cumplir con el enfoque del curso: contexto + datos + contraste, y para separar vistas técnicas, narrativas y de sandbox interactivo.

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

### Dashboard Dash local

La visualización ejecutiva vive en [SI7009/dashboard_app.py](SI7009/dashboard_app.py). Esa app tiene dos modos:

- Modo live: intenta leer métricas y parámetros desde MLflow y cargar el champion del registry.
- Modo offline: si MLflow no responde o el modelo no se puede cargar, usa métricas documentadas y un predictor heurístico para que la UI siga funcionando.

#### De dónde salen los datos cuando el modelo está offline

Cuando el registry no está accesible, el dashboard toma los valores de fallback definidos en el propio código y en la documentación del proyecto:

- Métricas ejecutivas: F2, recall, precision, Brier, flagging rate y estabilidad LOPO.
- Resumen de pipeline: estados de Bronze, Silver, Gold, ML y Viz.
- Comparaciones de modelos, calibración, drift y SHAP: tablas y figuras estáticas ya embebidas en la app.
- Predictor: si no puede cargar el champion, calcula una probabilidad heurística basada en los drivers principales del modelo.

#### Cómo “prender” el modelo

El dashboard no prende un modelo local por sí mismo; lo carga desde MLflow. Para que entre en modo live:

1. Verifica que el servidor MLflow esté disponible en `https://mlflow-server-919593201130.us-central1.run.app`.
2. Confirma que el registry `ShiftMetrics-DefectoEscapado` tenga el alias `champion` o `production`, o la versión `4`.
3. Asegúrate de tener acceso al proyecto de GCP y a los artefactos en `gs://shiftmetrics-bronze/mlruns/`.
4. Ejecuta la app con `python SI7009/dashboard_app.py`.

Si MLflow está visible pero el dashboard sigue en offline, normalmente significa que la app no pudo resolver uno de estos puntos: tracking URI, alias del registry, permisos, o artefactos del modelo.

#### Comandos útiles

```bash
python SI7009/dashboard_app.py
```

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://mlflow-server-919593201130.us-central1.run.app")
client = MlflowClient()
print(client.get_model_version_by_alias("ShiftMetrics-DefectoEscapado", "champion"))
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
