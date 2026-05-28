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

## Arquitectura del Pipeline (Medallion)

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

| Capa   | Bucket / Destino         | Descripción                              |
|--------|--------------------------|------------------------------------------|
| Bronze | `shiftmetrics-bronze`    | Datos crudos sin transformar             |
| Silver | `shiftmetrics-silver`    | Datos limpios, estandarizados, Parquet   |
| Gold   | `shiftmetrics_gold` (BQ) | Features y tablas analíticas para ML    |

---

## Bronze / Silver — Jobs PySpark

### Datasets Bronze

| Dataset        | Ruta GCS                  | Archivos   | Tamaño   | Formato  |
|----------------|---------------------------|------------|----------|----------|
| PROMISE        | `bronze/promise/`         | 144 CSV    | 534 MiB  | CSV      |
| Apache Jira    | `bronze/apache-jira/`     | 6 BSON.gz  | 1.9 GiB  | BSON.gz  |
| Red Hat Jira   | `bronze/redhat-jira/`     | 2 ZIP      | 266 MB   | ZIP/CSV  |
| GHArchive 2022 | `bronze/gharchive/`       | 24 JSON.gz | 1.7 GiB  | JSON.gz  |

**Estructura completa en GCS:**

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
├── gharchive/
│   └── [24 archivos .json.gz, 2022]
├── promise/
│   └── PROMISE-backup/bug-data/
│       └── [144 archivos CSV]
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

---

### Paso previo: Conversión BSON a Parquet (Apache Jira)

Apache Jira llegó como dump de MongoDB (`.bson.gz`). Spark y BigQuery no pueden leerlo directamente.

| Componente | Valor                                           |
|------------|-------------------------------------------------|
| Cluster    | `jira-converter` — single-node                  |
| Máquina    | n1-highmem-8 (8 vCPUs, 52 GB RAM)              |
| Script     | `gs://shiftmetrics-bronze/scripts/bson_fix.py`  |

**Lógica del script:**
1. Instala `pymongo` vía `subprocess.check_call()` al inicio del job.
2. Descarga el `.bson.gz` desde GCS.
3. Parsea BSON en streaming — lotes de 3,000 documentos.
4. Aplana documentos anidados hasta profundidad 2; arrays a string.
5. Escribe Parquet en chunks: primer chunk en `overwrite`, resto en `append`.
6. Sanitiza nombres de columnas (`.`, `-`, espacios → guiones bajos).

```python
def stream_bson_gz(gcs_path, chunk_size=3000):
    raw_bytes = blob.download_as_bytes()
    decompressed = gzip.decompress(raw_bytes)
    buf = io.BytesIO(decompressed)
    batch = []
    while True:
        size_bytes = buf.read(4)
        if len(size_bytes) < 4: break
        doc_size = struct.unpack("<i", size_bytes)[0]
        if doc_size < 5: break
        rest = buf.read(doc_size - 4)
        doc = bson.decode(size_bytes + rest)
        batch.append(flatten_doc(doc))
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

**Resultado:**

| Colección | BSON.gz  | Filas Parquet |
|-----------|----------|---------------|
| projects  | 1.2 MB   | pequeño       |
| users     | 7.8 MB   | pequeño       |
| worklogs  | 120 MB   | ~120K         |
| events    | 339 MB   | ~7.5M         |
| issues    | 729 MB   | ~978K         |
| comments  | 765 MB   | ~4.6M         |
| **TOTAL** | **1.9 GiB** | **~13M filas** |

**Comandos:**

```bash
gcloud dataproc clusters create jira-converter \
  --region=us-central1 --zone=us-central1-a \
  --master-machine-type=n1-highmem-8 \
  --master-boot-disk-size=100 \
  --single-node --image-version=2.1-debian11 \
  --project=shiftmetrics-analytics

gcloud dataproc jobs submit pyspark \
  gs://shiftmetrics-bronze/scripts/bson_fix.py \
  --cluster=jira-converter --region=us-central1 \
  --project=shiftmetrics-analytics
```

---

### Silver Job 01 — PROMISE

```
Input:  gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*/*.csv
Output: gs://shiftmetrics-silver/promise/
Script: silver_job_01_promise.py
```

**Schema:**
```
project, version, wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc,
dam, moa, mfa, cam, ic, cbm, amc,
bug_count (float), defect_flag (0/1), defect_density (float), total_modules
Partición: project
```

**Transformaciones:**
- Estandarizar columnas a lowercase.
- Castear métricas CK a float64.
- Binarizar `bug`: `(bug > 0).cast(IntegerType)` → `defect_flag`.
- Extraer `project` y `version` del nombre del archivo CSV.

**Cobertura:** 11 proyectos — ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces.

---

### Silver Job 02 — Apache Jira

```
Input:  gs://shiftmetrics-bronze/apache-jira-parquet/issues/
Output: gs://shiftmetrics-silver/apache-jira/
Script: silver_job_02_apache_jira.py
```

**Schema:**
```
key, project, issuetype_name, issue_category (bug/story/task/other),
status, resolution, created_ts, resolution_ts, cycle_time_days,
sprint (YYYY-M), bug_story_ratio (float)
Partición: project
```

**Transformaciones:**
- Usar columna `issuetype_name` (struct aplanado por `bson_fix.py` — no `issuetype`).
- Calcular `cycle_time_days = resolutiondate - created`.
- Extraer `project` de `key.split("-")[0]` (ej: `HADOOP-1234 → HADOOP`).
- Calcular `bug_story_ratio` por `(project, sprint)`.

---

### Silver Job 03 — Red Hat Jira

```
Input:  gs://shiftmetrics-bronze/redhat-jira-parquet/  (251 Parquets pre-generados)
Output: gs://shiftmetrics-silver/redhat-jira/
Script: silver_job_03_redhat_jira.py
```

Los 251 parquets fueron generados localmente desde `redhat-inputs.zip` y subidos a GCS antes de ejecutar el job. `redhat-outputs.zip` no contiene datos Jira (columnas: `Time, beta, alpha, epsilon, gamma, m`) — no se usa.

**Schema:**
```
issue_key, issue_type, issue_category (bug/story/task/other), status,
project_key, project_name, resolution, created_ts, resolved_ts,
cycle_time_days, sprint (YYYY-M),
num_bugs, num_stories, num_tasks, total_issues_sprint, bug_story_ratio (float)
Partición: project_key
```

**Transformaciones:**
- Parsear fechas con `dayfirst=True` — formato `DD/MM/YYYY HH:MM` — obligatorio.
- Calcular `cycle_time_days = Resolved - Created`.

**Cobertura:** 505,096 issues · 251 project keys.

---

### Silver Job 04 — GHArchive

```
Input:  gs://shiftmetrics-bronze/gharchive/*.json.gz (24 archivos)
Output: gs://shiftmetrics-silver/gharchive/
Script: silver_job_04_gharchive.py
```

**Schema:**
```
repo_name, apache_project_key (ej: "SPARK", "HADOOP"),
year, month,
push_count, deploy_frequency_weekly (push_count / 4.33),
total_prs_closed, prs_merged, prs_not_merged,
change_failure_rate (prs_not_merged / total_prs_closed)
Partición: year, month
```

**Transformaciones:**
- Filtrar `repo_name.startswith('apache/')` — obligatorio. Sin este filtro el CFR es 77.3% (incorrecto); con filtro: 26.1%.
- `apache_project_key = upper(split(repo_name, "/")[1])` — clave de join con Jira.

**Cobertura:** 4,994 eventos Apache 2022 · 12 meses.

---

## EDAS — Análisis Exploratorio

| Notebook           | Dataset                | Estado    | Resultado principal                       |
|--------------------|------------------------|-----------|-------------------------------------------|
| EDA_01_PROMISE     | PROMISE (144 CSV)      | Corregido | Schema incorrecto detectado y resuelto    |
| EDA_02_APACHE_JIRA | Apache Jira (Parquet)  | Corregido | Bug clasificador resuelto, 94 cols issues |
| EDA_03_REDHAT_JIRA | Red Hat Jira (ZIP/CSV) | Funcional | 505,096 issues, 251 proyectos validados   |
| EDA_04_GHARCHIVE   | GHArchive 2022         | Funcional | CFR real: 26.1%; 4,994 eventos Apache     |

**EDA_01 — PROMISE:** Los archivos en `AST_encoding/` tienen columnas nombradas como valores flotantes en lugar del schema CK clásico. La subcarpeta correcta es `bug-data/`. Columnas: `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc`. Target: `bug` binarizado con `(bug > 0) = 1`.

**EDA_02 — Apache Jira:** El clasificador de colecciones fallaba — todas quedaban como `other`. Corregido con regla directa `if coll_name == 'issues'`. Columnas a eliminar (>50% nulos): `regression`, `patchinfo`, `environment`, `duedate`, `timeestimate`.

**EDA_03 — Red Hat Jira:** Fechas en `DD/MM/YYYY HH:MM` se interpretaban mal sin `dayfirst=True`. Resultados: 505,096 issues, 86.4% con resolución, cycle time p50 = 28 días, p75 = 127 días, bug ratio = 1.21.

**EDA_04 — GHArchive:** CFR inicial de 77.3% calculado sin filtro sobre todos los repositorios del mundo. Con filtro `repo_name.startswith('apache/')` sobre los 24 archivos completos: CFR = 26.1%, 4,994 eventos Apache.

---

## Gold — Feature Store en BigQuery

### Estrategia de JOIN

```
Silver Apache Jira  (project, sprint)          ← BASE
    LEFT JOIN Silver PROMISE    ON project
    LEFT JOIN Silver GHArchive  ON apache_project_key=project AND year/month=sprint
    LEFT JOIN Silver Red Hat    ON project_key=project AND sprint=sprint
→   BigQuery: shiftmetrics_gold.sprint_features
```

| Fuente Silver  | Clave de unión                                       | Filas aportadas |
|----------------|------------------------------------------------------|-----------------|
| Apache Jira    | (project, sprint)                                    | 42,747          |
| Red Hat Jira   | project_key = project AND sprint = sprint            | 17,475          |
| GHArchive      | apache_project_key = project AND year/month = sprint | 205             |
| PROMISE CK     | project = project (por proyecto, no por sprint)      | 11 proyectos    |

### Schema: sprint_features

| Columna                   | Tipo    | Fuente     | Nullable | Descripción                                          |
|---------------------------|---------|------------|----------|------------------------------------------------------|
| `sprint_id`               | STRING  | Generado   | No       | `"{project}_{sprint}"` ej: `"SPARK_2022-3"`         |
| `project`                 | STRING  | Apache Jira| No       | Clave del proyecto                                   |
| `sprint`                  | STRING  | Apache Jira| No       | `"YYYY-M"`                                           |
| `avg_wmc`                 | FLOAT64 | PROMISE    | Si       | Weighted Methods per Class (promedio)                |
| `avg_dit`                 | FLOAT64 | PROMISE    | Si       | Depth of Inheritance Tree (promedio)                 |
| `avg_cbo`                 | FLOAT64 | PROMISE    | Si       | Coupling Between Objects (promedio)                  |
| `avg_rfc`                 | FLOAT64 | PROMISE    | Si       | Response for a Class (promedio)                      |
| `avg_lcom`                | FLOAT64 | PROMISE    | Si       | Lack of Cohesion of Methods (promedio)               |
| `avg_loc`                 | FLOAT64 | PROMISE    | Si       | Lines of Code (promedio)                             |
| `defect_density`          | FLOAT64 | PROMISE    | Si       | Modulos defectuosos / total modulos                  |
| `bug_story_ratio`         | FLOAT64 | Jira       | No       | Bugs / (Stories + Tasks) por sprint                  |
| `avg_cycle_time_days`     | FLOAT64 | Jira       | No       | Tiempo promedio de resolucion (dias)                 |
| `num_bugs_sprint`         | INT64   | Jira       | No       | Numero de bugs en el sprint                          |
| `num_stories_sprint`      | INT64   | Jira       | No       | Numero de stories en el sprint                       |
| `deploy_frequency_weekly` | FLOAT64 | GHArchive  | Si       | Pushes a main/master por semana                      |
| `change_failure_rate`     | FLOAT64 | GHArchive  | Si       | PRs sin merge / total PRs cerrados                   |
| `defecto_escapado`        | INT64   | Derivado   | No       | TARGET ML: 0 o 1 (avg_cycle_time_days > 30 dias)    |

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
|-------------|-----------|---------|---------------------|-----------------|
| 42,747      | 619       | 250     | 30,116              | 70.5%           |

**Nota sobre NULLs en CK:** Las columnas `avg_wmc`, `avg_cbo`, `avg_rfc`, `avg_lcom`, `avg_loc` y `defect_density` son NULL para 608 de los 619 proyectos. Solo los 11 proyectos de PROMISE tienen metricas CK. Para ML se recomienda agregar `has_ck_data` (1/0) e imputar por mediana.

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
