# ShiftMetrics Analytics — Análisis Completo del Proyecto
**Proyecto Integrador · Master Data Science & Analytics · EAFIT**
**Equipo SI7006 · Actualizado: 2026-05-22 (Gold completada)**

---

## Tabla de Contenidos

1. [Objetivo y Contexto](#1-objetivo-y-contexto)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Recursos GCP](#3-recursos-gcp)
4. [Estado Actual del Pipeline](#4-estado-actual-del-pipeline)
5. [Capa Bronze — Datasets de entrada](#5-capa-bronze--datasets-de-entrada)
6. [EDAs — Análisis Exploratorio por Dataset](#6-edas--análisis-exploratorio-por-dataset)
7. [Capa Silver — Jobs PySpark y Schemas](#7-capa-silver--jobs-pyspark-y-schemas)
8. [Capa Gold — sprint_features en BigQuery](#8-capa-gold--sprint_features-en-bigquery)
9. [Dataset Sintético](#9-dataset-sintético)
10. [Pipeline de ML](#10-pipeline-de-ml)
11. [Deploy](#11-deploy)
12. [Hoja de Ruta Priorizada](#12-hoja-de-ruta-priorizada)
13. [Estructura de Archivos del Repositorio](#13-estructura-de-archivos-del-repositorio)
14. [Advertencias y Hallazgos Críticos](#14-advertencias-y-hallazgos-críticos)
15. [Comandos GCP de Referencia](#15-comandos-gcp-de-referencia)

---

## 1. Objetivo y Contexto

### Problema
Predecir si un sprint de software producirá un **defecto escapado** — es decir, un bug que llega a producción sin ser detectado antes del release.

### Variable objetivo
```
defecto_escapado = 0 | 1   (por sprint)
```

### Enfoque ML
| Parámetro | Valor |
|---|---|
| Modelos | Logistic Regression (baseline) + XGBoost |
| Interpretabilidad | SHAP values |
| Métrica principal | **F2-Score** (penaliza más los falsos negativos) |
| Split de validación | **Temporal** 70% train / 15% val / 15% test |

La métrica F2-Score se elige porque el costo de no detectar un defecto escapado (falso negativo) es mayor que el de una falsa alarma.

---

## 2. Arquitectura del Sistema

```
[Bronze]          [Silver]            [Gold]              [ML]           [Deploy]
GCS raw data  →  Parquet limpio  →  BigQuery           →  Modelo     →  Cloud Run
gs://..bronze    Dataproc Spark     sprint_features       LR + XGBoost   API REST
                 gs://..silver      (una fila/sprint)      F2-Score
```

### Stack tecnológico
| Componente | Tecnología |
|---|---|
| Almacenamiento raw | Google Cloud Storage (GCS) |
| Procesamiento ETL | Dataproc (Apache Spark / PySpark) |
| Data Warehouse | BigQuery |
| Orquestación ML | Vertex AI |
| Serving | Cloud Run |
| EDAs locales | Python (pandas, matplotlib) |
| EDAs en nube | Google Colab |

---

## 3. Recursos GCP

| Recurso | Valor |
|---|---|
| Proyecto GCP | `shiftmetrics-analytics` |
| Project ID | `919593201130` |
| Service Account | `919593201130-compute@developer.gserviceaccount.com` |
| Región | `us-central1` |
| Zona | `us-central1-a` |
| Créditos disponibles | ~$275 USD |
| Bucket Bronze | `gs://shiftmetrics-bronze` (4.36 GiB, 176+ objetos) |
| Bucket Silver | `gs://shiftmetrics-silver` |
| Dataset BigQuery | `shiftmetrics-analytics:shiftmetrics_gold` *(a crear)* |
| Tabla BigQuery | `shiftmetrics_gold.sprint_features` *(a crear)* |
| Cluster Dataproc | `shiftmetrics-cluster` — **APAGADO** (recrear para Gold) |

### Permisos configurados
- Service account `919593201130-compute@developer.gserviceaccount.com` tiene `roles/storage.objectAdmin` en el proyecto.
- IAM en buckets de staging/temp de Dataproc ya configurado.

---

## 4. Estado Actual del Pipeline

```
Bronze ✅ COMPLETA  →  Silver ✅ COMPLETA  →  Gold ✅ COMPLETA  →  ML ⏳ SIGUIENTE  →  Deploy ⏳
```

| Capa | Estado | Detalle |
|---|---|---|
| **Bronze** | ✅ Completa | 4 datasets en GCS, sin cambios necesarios |
| **Silver** | ✅ Completa | 4 jobs escritos, ejecutados y verificados en Dataproc |
| **Gold** | ✅ Completa | `sprint_features` en BigQuery — 42,747 filas, 619 proyectos, prevalencia 70.5% |
| **ML Pipeline** | ⏳ **SIGUIENTE PASO** | Requiere decisión sobre prevalencia + imputación CK |
| **Deploy Cloud Run** | ⏳ Pendiente | Requiere ML |

---

## 5. Capa Bronze — Datasets de entrada

| # | Dataset | Ruta GCS | Volumen | Notas |
|---|---|---|---|---|
| 1 | PROMISE (métricas CK) | `bronze/promise/PROMISE-backup/bug-data/` | ~20 K módulos Java | CSVs por proyecto en subcarpetas |
| 2 | Apache Jira | `bronze/apache-jira-parquet/` | ~13 M filas | 6 colecciones BSON convertidas a Parquet |
| 3 | Red Hat Jira | `bronze/redhat-jira/redhat-inputs.zip` | 505,096 issues | 251 proyectos · 251 CSVs |
| 4 | GHArchive 2022 | `bronze/gharchive/*.json.gz` | 24 archivos · 1.69 GiB | Rango 2022-01-01 → 2022-03-15 |
| 5 | Sintético (SimPy+Faker) | *(a generar)* | ~2 000 sprints | Calibrado con EDAs |

### Colecciones Apache Jira (6)
`issues`, `events`, `comments`, `worklogs`, `users`, `sprints`

> Para Silver solo se usan **`issues`** y **`events`** (el resto se descarta).

---

## 6. EDAs — Análisis Exploratorio por Dataset

### 6.1 EDA_01 — PROMISE (`EDA_01_PROMISE_corregido.ipynb`)

**Plataforma:** Google Colab  
**Estado:** ✅ Corregido y funcional

#### Correcciones aplicadas
| Error original | Corrección |
|---|---|
| `[m.lower() for m in]` (sintaxis rota) | `[m.lower() for m in CK_METRICS]` |
| Leía carpeta `AST_encoding/` | Corregido a `bug-data/` |
| Variable `hal_cols` | Renombrada a `ck_cols` |
| Carga parcial (10 CSVs) | Carga completa de todos los CSVs |

#### Hallazgos clave
- **Path real:** `promise/PROMISE-backup/bug-data/*/*.csv` (CSVs en subcarpetas por proyecto, no en raíz)
- **18 métricas CK disponibles:** `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc`
- **Target:** columna `bug` → binarizar: `(bug > 0) = 1` → `defect_flag`
- **Sin nulos** en `bug-data/`
- **Proyectos:** ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces (11 total)

#### Parámetros para Silver
```python
INPUT_PATH  = "gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*/*.csv"
OUTPUT_PATH = "gs://shiftmetrics-silver/promise/"
CK_METRICS  = ['wmc','dit','noc','cbo','rfc','lcom','ca','ce','npm',
                'lcom3','loc','dam','moa','mfa','cam','ic','cbm','amc']
TARGET      = 'bug'          # binarizar → defect_flag
PARTITION   = 'project'
```

---

### 6.2 EDA_02 — Apache Jira (`EDA_02_APACHE_JIRA_corregido.ipynb`)

**Plataforma:** Google Colab  
**Estado:** ✅ Corregido y funcional

#### Correcciones aplicadas
| Error original | Corrección |
|---|---|
| Clasificador de colecciones roto | Regla manual `if coll_name == 'issues'` |
| Transiciones en colección separada | Extraídas de `events.items` con `field == 'status'` |
| Cargaba solo 1 parquet por colección | Ahora carga hasta 5 (`SAMPLE_FILES = 5`) |
| Sin métricas de sprint | Añadidos Cycle Time, Bug-to-Story Ratio, extracción de `project` |

#### Hallazgos clave
- **Colecciones útiles:** `issues`, `events` → descartar `comments`, `worklogs`, `users`
- **Cycle Time:** `resolutiondate − created` en días
- **Extracción de proyecto:** `key.split('-')[0]` → `HADOOP-1234 → HADOOP`
- **Columna de tipo:** `issuetype_name` (struct aplanado por `bson_to_parquet.py`) — **NO `issuetype`**
- **Cols a eliminar (>50% nulos):** `regression`, `patchinfo`, `environment`, `duedate`, `timeestimate`
- **Bug-to-Story Ratio:** `bugs / (stories + tasks)` por proyecto y sprint

#### Parámetros para Silver
```python
INPUT_ISSUES = "gs://shiftmetrics-bronze/apache-jira-parquet/issues/"
INPUT_EVENTS = "gs://shiftmetrics-bronze/apache-jira-parquet/events/"
OUTPUT_PATH  = "gs://shiftmetrics-silver/apache-jira/"
TYPE_COL     = 'issuetype_name'   # ← columna real tras aplanar BSON
CYCLE_TIME   = 'resolutiondate - created'
SPRINT_COL   = 'YYYY-M'          # aproximación de sprint por fecha
PARTITION    = 'project'
```

---

### 6.3 EDA_03 — Red Hat Jira (`EDA_03_REDHAT_JIRA.ipynb`)

**Plataforma:** Local (Colab crasheaba por OOM)  
**Estado:** ✅ Ejecutado localmente con `MODE = 'local'`

#### Correcciones aplicadas
| # | Error original | Corrección |
|---|---|---|
| 1 | `zip_blobs` no definida | Guard `if 'zip_blobs' not in locals()` |
| 2 | Descarga parcial del ZIP | `tempfile.NamedTemporaryFile` (descarga completa) |
| 3 | Solo procesaba algunos CSVs | Itera todos los CSVs de ambos ZIPs |
| 4 | Fechas mal interpretadas | `dayfirst=True` en todos los `pd.to_datetime()` |
| 5 | `MODE='local'` dejaba `dfs=[]` | **Nueva celda 3b** que lee parquets generados |
| 6 | `AttributeError` en issue types | `'Issue Type'` añadido a `type_candidates` |
| 7 | Cycle Time sin `dayfirst=True` | Corregido |

#### Hallazgo crítico — `redhat-outputs.zip` NO ES JIRA
> **`redhat-outputs.zip` (248 MB) contiene columnas `['Time', 'beta', 'alpha', 'epsilon', 'gamma', 'm']`** — son datos de un modelo matemático/simulación, NO issues de Jira. **Nunca usar en el pipeline.**

#### Dataset real — 505,096 issues
**Schema:** `Issue key | Issue Type | Status | Project key | Project name | Project type | Resolution | Created | Resolved`

**Formato de fechas:** `DD/MM/YYYY HH:MM` (ej: `22/03/2023 17:04`) → **`dayfirst=True` OBLIGATORIO**

#### Distribución de Issue Types
| Tipo | Issues |
|---|---|
| Bug | 223,922 |
| Task | 112,917 |
| Story | 71,406 |
| Feature Request | 33,654 |
| Enhancement | 24,826 |
| Epic | 13,293 |
| Component Upgrade | 9,650 |
| Feature | 6,656 |
| Spike | 3,013 |
| Release | 1,104 |

#### Distribución de Estados
| Estado | Issues |
|---|---|
| Closed | 362,387 |
| Resolved | 61,483 |
| New | 21,621 |
| Open | 13,427 |
| To Do | 8,453 |
| Done | 6,696 |

#### Cycle Time (Created → Resolved)
| Métrica | Valor |
|---|---|
| Issues con resolución | 436,475 / 505,096 **(86.4%)** |
| **p50** | **28.0 días** |
| **p75** | **127.0 días** |
| p90 | 400.0 días |
| Media | 156.5 días |

#### Nulos por columna
| Columna | % Nulos | Acción |
|---|---|---|
| Fix Version/s | 99.96% | **Eliminar** |
| Assignee | 99.95% | **Eliminar** |
| Updated | 99.91% | **Eliminar** |
| Resolved | 13.56% | Mantener (issues sin cerrar) |
| Resolution | 13.39% | Mantener |
| Created | 0.09% | Mantener |

#### Métricas clave para simulador
```python
cycle_time_p50       = 28       # días
cycle_time_p75       = 127      # días
cycle_time_mean      = 156.5    # días
bug_ratio            = 1.21     # bugs / (tasks + stories) = 223922 / (112917 + 71406)
issues_resueltos_pct = 0.864    # 86.4%
n_proyectos          = 251
```

#### Cómo ejecutar (local)
1. Celdas 0–5: ya tienen outputs válidos — **no re-ejecutar**
2. Celda 6 (carga ZIPs): ya escribió parquets en `redhat_parquet/` — **no re-ejecutar**
3. **Celda 3b en adelante:** ejecutar secuencialmente — lee parquets y corre el análisis

---

### 6.4 EDA_04 — GHArchive (`EDA_04_GHARCHIVE.ipynb`)

**Plataforma:** Local  
**Estado:** ✅ Ejecutado localmente (24/24 archivos)

#### Correcciones aplicadas
| # | Error original | Corrección |
|---|---|---|
| 1 | CFR calculado sin filtro Apache | Filtro `repo_name.startswith('apache/')` en tiempo de carga |
| 2 | Solo 2 archivos / 10K eventos (código viejo) | Itera los 24 archivos completos |
| 3 | `actor_login` capturaba usernames con "apache" | **Eliminado** — solo se usa `repo_name` |
| 4 | `AttributeError: DatetimeIndex has no .dt` | `pd.Series(pd.to_datetime(...))` |

#### Resultados validados
| Métrica | Valor |
|---|---|
| Archivos procesados | 24 de 24 (1.69 GiB total) |
| Eventos Apache cargados | 4,994 |
| Rango temporal | 2022-01-01 → 2022-03-15 |
| **Change Failure Rate** | **26.1%** (PRs cerrados sin merge / total PRs) |
| CFR sin filtro (incorrecto) | ~~77.3%~~ |
| Evento dominante | `IssueCommentEvent` (21.9%) |
| Repos más activos | airflow, spark, arrow, pulsar, beam |
| Deploy Freq top | `apache/camel` (11 pushes), `apache/airflow` (10 pushes) |

#### Clave de join con Jira
```python
# En Silver 04
apache_project_key = upper(split(repo_name, "/")[1])
# "apache/spark" → "SPARK"
# "apache/hadoop" → "HADOOP"
# Coincide con project key de Apache Jira
```

#### Parámetros para Silver
```python
INPUT_PATH       = "gs://shiftmetrics-bronze/gharchive/*.json.gz"
OUTPUT_PATH      = "gs://shiftmetrics-silver/gharchive/"
REPO_FILTER      = "apache/"    # OBLIGATORIO
EVENT_TYPES      = ['PushEvent', 'PullRequestEvent']
DEPLOY_FREQ      = "push_count / 4.33"   # pushes/semana a main/master
CFR              = "prs_not_merged / total_prs_closed"
PARTITION        = ['year', 'month']
```

---

## 7. Capa Silver — Jobs PySpark y Schemas

Todos los scripts están en `gs://shiftmetrics-bronze/scripts/`. Outputs verificados en `gs://shiftmetrics-silver/`.

### 7.1 Jobs ejecutados

| Job | Script | Output | Partición | Estado |
|---|---|---|---|---|
| 01 PROMISE | `silver_job_01_promise.py` | `gs://shiftmetrics-silver/promise/` | `project` | ✅ |
| 02 Apache Jira | `silver_job_02_apache_jira.py` | `gs://shiftmetrics-silver/apache-jira/` | `project` | ✅ |
| 03 Red Hat Jira | `silver_job_03_redhat_jira.py` | `gs://shiftmetrics-silver/redhat-jira/` | `project_key` (251) | ✅ |
| 04 GHArchive | `silver_job_04_gharchive.py` | `gs://shiftmetrics-silver/gharchive/` | `year`, `month` | ✅ |

### 7.2 Correcciones aplicadas durante ejecución

| Job | Problema | Corrección |
|---|---|---|
| 01 | Path `bug-data/*.csv` no encontraba archivos | Corregido a `bug-data/*/*.csv` |
| 02 | Path `apache-jira-parquet/*.parquet` incorrecto | Corregido a `apache-jira-parquet/issues/` |
| 02 | Columna `issuetype` no existía | Cambiado a `issuetype_name` (struct aplanado) |
| 03 | Spark no puede leer ZIPs directamente | 251 parquets pre-subidos a `gs://shiftmetrics-bronze/redhat-jira-parquet/` |
| 04 | Sin correcciones | Corrió limpio |

### 7.3 Schema Silver 01 — PROMISE
```
project          STRING    -- ant, camel, jedit, log4j, lucene, poi, ...
version          STRING    -- versión del proyecto
wmc              FLOAT64   -- Weighted Methods per Class
dit              FLOAT64   -- Depth of Inheritance Tree
noc              FLOAT64   -- Number of Children
cbo              FLOAT64   -- Coupling Between Objects
rfc              FLOAT64   -- Response For a Class
lcom             FLOAT64   -- Lack of Cohesion of Methods
ca               FLOAT64   -- Afferent Coupling
ce               FLOAT64   -- Efferent Coupling
npm              FLOAT64   -- Number of Public Methods
lcom3            FLOAT64
loc              FLOAT64   -- Lines of Code
dam              FLOAT64
moa              FLOAT64
mfa              FLOAT64
cam              FLOAT64
ic               FLOAT64
cbm              FLOAT64
amc              FLOAT64
bug_count        FLOAT64   -- valor original de la columna 'bug'
defect_flag      INT64     -- (bug > 0) ? 1 : 0
defect_density   FLOAT64   -- módulos con defecto / total módulos del proyecto-versión
total_modules    INT64
-- Partición: project
```

**Valores esperados:** 11 proyectos (ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces)

### 7.4 Schema Silver 02 — Apache Jira
```
key              STRING    -- ej: "HADOOP-1234"
project          STRING    -- ej: "HADOOP"
issuetype_name   STRING    -- ← columna real (struct aplanado por bson_to_parquet.py)
issue_category   STRING    -- bug | story | task | other
status           STRING
resolution       STRING
created_ts       TIMESTAMP
resolution_ts    TIMESTAMP
cycle_time_days  FLOAT64
sprint           STRING    -- "YYYY-M" (aproximación por fecha de creación)
bug_story_ratio  FLOAT64
-- Partición: project
```

**Nota:** La columna de tipo se llama **`issuetype_name`** (no `issuetype`) porque el documento BSON original fue aplanado por `bson_to_parquet.py`.

### 7.5 Schema Silver 03 — Red Hat Jira
```
issue_key             STRING    -- ej: "AAH-123"
issue_type            STRING    -- Bug | Task | Story | ...
issue_category        STRING    -- bug | story | task | other
status                STRING    -- Closed | Resolved | New | ...
project_key           STRING    -- ej: "AAH"
project_name          STRING
resolution            STRING
created_ts            TIMESTAMP -- parseado con dayfirst=True
resolved_ts           TIMESTAMP -- parseado con dayfirst=True
cycle_time_days       FLOAT64
sprint                STRING    -- "YYYY-M"
num_bugs              INT64
num_stories           INT64
num_tasks             INT64
total_issues_sprint   INT64
bug_story_ratio       FLOAT64
-- Partición: project_key (251 particiones)
```

### 7.6 Schema Silver 04 — GHArchive
```
repo_name                STRING    -- "apache/spark"
apache_project_key       STRING    -- "SPARK" (clave de join con Jira)
year                     INT64
month                    INT64
push_count               INT64
deploy_frequency_weekly  FLOAT64   -- push_count / 4.33
total_prs_closed         INT64
prs_merged               INT64
prs_not_merged           INT64
change_failure_rate      FLOAT64   -- prs_not_merged / total_prs_closed
-- Partición: year, month
```

---

## 8. Capa Gold — sprint_features en BigQuery ✅ COMPLETA

### Ejecución — Paso a paso

| Paso | Acción | Resultado |
|---|---|---|
| 1 | Verificar Silver | 4 tablas presentes con datos (apache-jira, redhat-jira, gharchive, promise) ✅ |
| 2 | Confirmar dataset BigQuery | `shiftmetrics_gold` ya existía ✅ |
| 3 | Recrear cluster Dataproc | `shiftmetrics-cluster` n1-standard-2 · 2 CPUs/nodo × 3 = 6 CPUs (cuota: 8) ✅ |
| 4 | Subir script a GCS | `gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py` ✅ |
| 5 | Ejecutar Gold job | Job completado sin errores ✅ |
| 6 | Verificar tabla BigQuery | 42,747 filas confirmadas ✅ |
| 7 | Eliminar cluster | Eliminado para conservar créditos ✅ |

> **Nota técnica:** El flag `--jars` aplica al *enviar jobs*, no al crear clusters. Dataproc incluye el conector BigQuery por defecto — el job corrió **sin necesidad de `--jars`**.

### Resultados verificados en BigQuery

| Métrica | Valor |
|---|---|
| Total filas | **42,747** |
| Proyectos distintos | **619** |
| Sprints distintos | **250** |
| Defectos positivos (`defecto_escapado=1`) | **30,116** |
| Prevalencia | **70.5%** |

### Desglose de filas por fuente Silver

| Fuente Silver | Aporte |
|---|---|
| Apache Jira | 42,747 (base del JOIN) |
| Red Hat Jira | 17,475 filas joined |
| GHArchive | 205 filas joined |
| PROMISE (CK metrics) | 11 proyectos joined |

### Consideraciones para ML

**1. Prevalencia alta (70.5%)**

La variable target `defecto_escapado=1` aparece en el 70.5% de los sprints — esto es viable pero puede sesgar el modelo. Opciones:

| Estrategia | Descripción | Cuándo usar |
|---|---|---|
| Undersampling | Reducir clase mayoritaria (defecto=1) hasta 60/40 o 50/50 | Si hay suficientes filas post-filtro |
| Ajustar umbral `cycle_time_days` | Subir de 30 → 60 días para marcar defecto | Si la definición de "defecto" es negociable |
| `scale_pos_weight` en XGBoost | Parámetro nativo para clases desbalanceadas | Si se quiere usar los datos tal cual |
| Sin cambio + F2-Score | F2 ya penaliza falsos negativos — puede ser suficiente | Opción conservadora |

**2. NULLs en métricas CK (avg_wmc, avg_cbo, avg_rfc, etc.)**

Las métricas CK son NULL para proyectos que no están en PROMISE (la mayoría de Red Hat y muchos Apache). Opciones:

| Estrategia | Descripción |
|---|---|
| Imputar por mediana | Reemplazar NULL con mediana de cada columna CK |
| Filtrar proyectos completos | Usar solo proyectos Apache con datos PROMISE + Jira + GHArchive |
| Entrenar sin CK features | Excluir columnas CK del modelo si cobertura < 30% |

### Objetivo
Una fila por `(project, sprint)` que une las 4 fuentes Silver para entrenar el modelo ML.

### Tabla destino
`shiftmetrics-analytics:shiftmetrics_gold.sprint_features`

### Schema completo
```sql
-- Identificación del sprint
sprint_id              STRING     -- "{project}_{sprint}" ej: "SPARK_2022-3"
project                STRING     -- clave del proyecto, ej: "SPARK"
sprint                 STRING     -- "YYYY-M", ej: "2022-3"

-- CK Metrics (de Silver PROMISE — promedio por proyecto/versión)
avg_wmc                FLOAT64    -- Weighted Methods per Class
avg_dit                FLOAT64    -- Depth of Inheritance Tree
avg_cbo                FLOAT64    -- Coupling Between Objects
avg_rfc                FLOAT64    -- Response For a Class
avg_lcom               FLOAT64    -- Lack of Cohesion of Methods
avg_loc                FLOAT64    -- Lines of Code
defect_density         FLOAT64    -- módulos defectuosos / total módulos

-- Jira Metrics (Apache Jira o Red Hat Jira)
bug_story_ratio        FLOAT64    -- bugs / (tasks + stories)
avg_cycle_time_days    FLOAT64    -- tiempo promedio de resolución
num_bugs_sprint        INT64      -- cantidad de bugs en el sprint
num_stories_sprint     INT64      -- cantidad de stories en el sprint

-- DORA Metrics (GHArchive)
deploy_frequency_weekly FLOAT64   -- pushes/semana a main/master
change_failure_rate    FLOAT64    -- PRs sin merge / total PRs closed

-- Target
defecto_escapado       INT64      -- 0 o 1 (variable objetivo ML)
```

### Estrategia de JOIN
```
Silver Apache Jira  (base — project, sprint)
    LEFT JOIN Silver PROMISE
        ON silver_apache.project = silver_promise.project
        -- CK metrics son por proyecto, no por sprint individual

    LEFT JOIN Silver GHArchive
        ON silver_apache.project = silver_gh.apache_project_key
       AND silver_apache.year    = silver_gh.year
       AND silver_apache.month   = silver_gh.month

    LEFT JOIN Silver Red Hat
        ON silver_apache.project = silver_rh.project_key
       AND silver_apache.sprint  = silver_rh.sprint

→ BigQuery: shiftmetrics_gold.sprint_features
```

**Clave de join GHArchive:**
- `silver_gh.apache_project_key` = `upper(split(repo_name, "/")[1])`
- Ejemplo: `"apache/spark"` → `"SPARK"` ↔ coincide con Apache Jira `project = "SPARK"`

### Pasos para construir Gold

1. Crear dataset en BigQuery:
   ```bash
   bq mk --dataset shiftmetrics-analytics:shiftmetrics_gold
   ```

2. Recrear cluster Dataproc **con jar BigQuery**:
   ```bash
   gcloud dataproc clusters create shiftmetrics-cluster \
     --region=us-central1 \
     --zone=us-central1-a \
     --master-machine-type=n1-standard-4 \
     --worker-machine-type=n1-standard-4 \
     --num-workers=2 \
     --master-boot-disk-size=50GB \
     --worker-boot-disk-size=50GB \
     --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
     --project=shiftmetrics-analytics
   ```

3. Escribir `gold_job_sprint_features.py` (PySpark que lee los 4 Silver y hace el JOIN)

4. Ejecutar el Gold job:
   ```bash
   gcloud dataproc jobs submit pyspark \
     gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py \
     --cluster=shiftmetrics-cluster \
     --region=us-central1 \
     --project=shiftmetrics-analytics
   ```

5. Verificar en BigQuery Console:
   ```sql
   SELECT COUNT(*), COUNT(DISTINCT project), COUNT(DISTINCT sprint)
   FROM shiftmetrics_gold.sprint_features;
   ```

### Verificación de Silver antes de Gold
```bash
gsutil ls gs://shiftmetrics-silver/promise/       # 11 proyectos
gsutil ls gs://shiftmetrics-silver/apache-jira/   # HADOOP, SPARK, HBASE, HDFS, HIVE, ...
gsutil ls gs://shiftmetrics-silver/redhat-jira/   # 251 project_keys
gsutil ls gs://shiftmetrics-silver/gharchive/     # year=2022/month=1/ ... month=12/
```

---

## 9. Dataset Sintético

### Objetivo
Generar ~2,000 sprints de datos sintéticos con **SimPy + Faker**, calibrados a partir de los EDAs reales, para aumentar el tamaño del dataset de entrenamiento.

### Parámetros calibrados

| Parámetro | Valor | Fuente |
|---|---|---|
| `cycle_time_p50` | 28 días | EDA_03 Red Hat Jira |
| `cycle_time_p75` | 127 días | EDA_03 Red Hat Jira |
| `bug_story_ratio_rh` | 1.21 | EDA_03 Red Hat Jira |
| `cfr_proxy` | 0.261 | EDA_04 GHArchive Apache |
| `issues_resueltos_pct` | 0.864 | EDA_03 Red Hat Jira |
| `n_sprints` | 2,000 | Estimado para ML |

### Distribuciones sugeridas
```python
import numpy as np
from scipy.stats import beta

n_sprints = 2_000

# Cycle Time — lognormal calibrada para p50=28d
avg_cycle_time = np.random.lognormal(mean=3.3, sigma=1.2, size=n_sprints)

# Bug/Story Ratio — lognormal
bug_story_ratio = np.random.lognormal(mean=0.18, sigma=0.6, size=n_sprints)

# Change Failure Rate — beta con media≈0.26
change_failure_rate = np.random.beta(a=2.6, b=7.4, size=n_sprints)

# Deploy Frequency — Poisson
deploy_frequency = np.random.poisson(lam=6, size=n_sprints)  # pushes/semana
```

### Estado
Parámetros listos. La implementación se hace después de validar la capa Gold.

---

## 10. Pipeline de ML

### Flujo
```
sprint_features (BigQuery)
    → Feature engineering
    → Split temporal 70/15/15
    → Baseline: Logistic Regression
    → Modelo final: XGBoost
    → Interpretabilidad: SHAP values
    → Evaluación: F2-Score
    → Serialización del modelo
```

### Decisiones de diseño

| Decisión | Razón |
|---|---|
| Split **temporal** (no aleatorio) | Evita data leakage futuro → pasado |
| **F2-Score** como métrica | Penaliza más los falsos negativos (defecto que escapa sin detectarse) |
| **LR como baseline** | Interpretable, establece referencia |
| **XGBoost como modelo final** | Maneja bien features heterogéneos y no-linealidades |
| **SHAP** | Explicabilidad por feature y por sprint individual |

### Variables esperadas
- **Features numéricas:** avg_wmc, avg_dit, avg_cbo, avg_rfc, avg_lcom, avg_loc, defect_density, bug_story_ratio, avg_cycle_time_days, num_bugs_sprint, num_stories_sprint, deploy_frequency_weekly, change_failure_rate
- **Target:** `defecto_escapado` (0/1)

---

## 11. Deploy

### API REST en Cloud Run
- Endpoint: `POST /predict`
- Input: métricas del sprint actual (JSON)
- Output: `{"defecto_escapado": 0|1, "probability": float, "shap_values": {...}}`
- Runtime: Python / FastAPI
- Modelo: cargado desde GCS o Vertex AI Model Registry

---

## 12. Hoja de Ruta Priorizada

| # | Tarea | Estado |
|---|---|---|
| 1 | Silver Job 01 — PROMISE | ✅ Ejecutado y verificado |
| 2 | Silver Job 02 — Apache Jira | ✅ Ejecutado y verificado |
| 3 | Silver Job 03 — Red Hat Jira | ✅ Ejecutado y verificado |
| 4 | Silver Job 04 — GHArchive | ✅ Ejecutado y verificado |
| 5 | Crear dataset BigQuery `shiftmetrics_gold` | ✅ Ya existía — confirmado |
| 6 | Escribir Gold job PySpark (join 4 Silver → sprint_features) | ✅ Ejecutado |
| 7 | Verificar tabla sprint_features en BigQuery | ✅ 42,747 filas · 619 proyectos |
| **8** | **Decisión sobre prevalencia (70.5%) — undersample o ajuste umbral** | **⏳ SIGUIENTE** |
| **9** | **ML Pipeline — LR baseline + XGBoost + SHAP (F2-Score)** | **⏳ SIGUIENTE** |
| 10 | Dataset Sintético — SimPy + Faker (~2,000 sprints) | ⏳ Parámetros listos |
| 11 | Deploy Cloud Run (API REST de predicción) | ⏳ Requiere ML |

---

## 13. Estructura de Archivos del Repositorio

```
Shiftmetrics-bronze-silver/
│
├── SHIFTMETRICS_ANALISIS_COMPLETO.md        ← Este archivo
│
├── Context/
│   ├── ShiftMetrics_Contexto_y_Hoja_de_Ruta.md   ← Documento maestro (actualizado 2026-05-22)
│   ├── CONTEXTO_SESION_2026_05_21.md              ← Bitácora sesión 21/05 (Bronze + EDAs)
│   └── CONTEXTO_SESION_2026_05_22.md              ← Handoff sesión 22/05 (Silver verificada)
│
├── EDA_03_OUTPUTS.md                         ← Outputs validados EDA Red Hat
│
├── EDAS/
│   ├── EDA_01_PROMISE_corregido.ipynb        ← ✅ Colab — CK metrics, target bug
│   ├── EDA_02_APACHE_JIRA_corregido.ipynb    ← ✅ Colab — cycle time, issue types
│   ├── EDA_03_REDHAT_JIRA.ipynb              ← ✅ Local — 505K issues, redhat_parquet/
│   └── EDA_04_GHARCHIVE.ipynb                ← ✅ Local — DORA metrics, CFR=26.1%
│
└── Jobs/
    ├── silver_job_01_promise.py              ← ✅ PySpark Bronze→Silver PROMISE
    ├── silver_job_02_apache_jira.py          ← ✅ PySpark Bronze→Silver Apache Jira
    ├── silver_job_03_redhat_jira.py          ← ✅ PySpark Bronze→Silver Red Hat Jira
    └── silver_job_04_gharchive.py            ← ✅ PySpark Bronze→Silver GHArchive

-- Datos en GCS (no en este repo) --
gs://shiftmetrics-bronze/                    ← 4.36 GiB datos crudos
gs://shiftmetrics-silver/                    ← 4 datasets Parquet limpios

-- Datos locales (ignorados por .gitignore) --
redhat_parquet/                              ← 251 parquets (ya subidos a GCS)
```

---

## 14. Advertencias y Hallazgos Críticos

### Crítico — No usar `redhat-outputs.zip`
> **`redhat-outputs.zip` NO contiene issues de Jira.**
> Columnas: `['Time', 'beta', 'alpha', 'epsilon', 'gamma', 'm']` — es un modelo matemático.
> **Usar ÚNICAMENTE `redhat-inputs.zip`** (251 CSVs, 505,096 issues reales).

### Crítico — Fechas Red Hat con `dayfirst=True`
```python
pd.to_datetime(df['Created'], dayfirst=True)
pd.to_datetime(df['Resolved'], dayfirst=True)
# Formato: DD/MM/YYYY HH:MM → ej "22/03/2023 17:04"
# Sin dayfirst=True los días y meses se invierten → Cycle Time completamente erróneo
```

### Crítico — Filtro `apache/` en GHArchive
```python
df = df[df['repo_name'].str.startswith('apache/')]
# Sin filtro: CFR = 77.3% (todos los repos del mundo) ← INCORRECTO
# Con filtro: CFR = 26.1% (solo proyectos Apache)    ← CORRECTO
```

### Crítico — Columna de tipo en Apache Jira
```python
# INCORRECTO:
df['issuetype']
# CORRECTO:
df['issuetype_name']   # struct aplanado por bson_to_parquet.py
```

### Advertencia — Path PROMISE
```
# INCORRECTO:
gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*.csv
# CORRECTO:
gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*/*.csv
# Los CSVs están en subcarpetas: bug-data/ant/*.csv, bug-data/camel/*.csv, etc.
```

### Advertencia — Red Hat Jira necesita pre-procesamiento
Spark no puede leer ZIPs directamente. Los 251 parquets ya están en:
```
gs://shiftmetrics-bronze/redhat-jira-parquet/
```
Si se re-ejecuta Silver 03, verificar que este path exista.

### Advertencia — Cluster Dataproc apagado
Para Gold se necesita recrear el cluster **con el jar del conector BigQuery**:
```bash
--jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar
```
Sin este jar el job Gold fallará al intentar escribir en BigQuery.

---

## 15. Comandos GCP de Referencia

### Crear cluster Dataproc (Gold)
```bash
gcloud dataproc clusters create shiftmetrics-cluster \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-4 \
  --num-workers=2 \
  --master-boot-disk-size=50GB \
  --worker-boot-disk-size=50GB \
  --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
  --project=shiftmetrics-analytics
```

### Crear dataset BigQuery
```bash
bq mk --dataset shiftmetrics-analytics:shiftmetrics_gold
```

### Subir script a GCS
```bash
gsutil cp gold_job_sprint_features.py \
  gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py
```

### Ejecutar job PySpark en Dataproc
```bash
gcloud dataproc jobs submit pyspark \
  gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py \
  --cluster=shiftmetrics-cluster \
  --region=us-central1 \
  --project=shiftmetrics-analytics
```

### Verificar Silver antes de Gold
```bash
gsutil ls gs://shiftmetrics-silver/promise/
gsutil ls gs://shiftmetrics-silver/apache-jira/
gsutil ls gs://shiftmetrics-silver/redhat-jira/
gsutil ls gs://shiftmetrics-silver/gharchive/
```

### Apagar cluster cuando no se use (ahorra créditos)
```bash
gcloud dataproc clusters delete shiftmetrics-cluster \
  --region=us-central1 --project=shiftmetrics-analytics
```

---

*ShiftMetrics Analytics · EAFIT SI7006 · Actualizado 2026-05-22 (Gold completada)*
