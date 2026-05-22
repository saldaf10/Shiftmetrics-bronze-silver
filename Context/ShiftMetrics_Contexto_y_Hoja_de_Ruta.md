# ShiftMetrics Analytics — Contexto General y Hoja de Ruta
**Proyecto Integrador — Master Data Science & Analytics — EAFIT**
**Equipo SI7006 · Actualizado: 2026-05-22 (Gold completada)**

---

## 1. Objetivo del Proyecto

Predecir **defecto escapado** (`defecto_escapado` = 0/1) en sprints de software usando ML (Logistic Regression + XGBoost), con validación temporal 70/15/15 y métrica F2-Score.

### Arquitectura: Medallion en GCP
```
Bronze (GCS raw)  →  Silver (Parquet limpio)  →  Gold (BigQuery sprint_features)  →  ML
```

### GCP
- **Proyecto**: `shiftmetrics-analytics` (ID: 919593201130)
- **Bucket Bronze**: `gs://shiftmetrics-bronze` — 4.36 GiB, 176+ objetos
- **Bucket Silver** (a poblar): `gs://shiftmetrics-silver`
- **Tabla Gold** (a crear): `shiftmetrics-analytics:gold.sprint_features`
- **Créditos disponibles**: ~$275 USD
- **Stack**: Cloud Storage + Dataproc (Spark) + BigQuery + Cloud Run + Vertex AI

---

## 2. Estado Actual por Capa

| Capa | Estado | Detalle |
|---|---|---|
| **Bronze** | ✅ Completa | 4 datasets en GCS, sin cambios necesarios |
| **Silver** | ✅ Completa | 4 jobs escritos, ejecutados y verificados en Dataproc |
| **Gold** | ✅ Completa | `sprint_features` en BigQuery — 42,747 filas, 619 proyectos, prevalencia 70.5% |
| **ML Pipeline** | ⏳ **SIGUIENTE PASO** | Requiere ajuste de prevalencia + imputación CK |
| **Deploy Cloud Run** | ⏳ Pendiente | Requiere ML |

---

## 3. Capa Bronze — Datasets ✅

| # | Dataset | Ruta GCS | Filas reales | Estado |
|---|---|---|---|---|
| 1 | PROMISE (métricas CK) | `bronze/promise/PROMISE-backup/bug-data/*.csv` | ~20K módulos | ✅ |
| 2 | Apache Jira | `bronze/apache-jira-parquet/` (6 colecciones) | ~13M filas | ✅ |
| 3 | Red Hat Jira | `bronze/redhat-jira/redhat-inputs.zip` | **505,096 issues** | ✅ |
| 4 | GHArchive 2022 | `bronze/gharchive/*.json.gz` (24 archivos) | 1.69 GiB | ✅ |
| 5 | Sintético (SimPy+Faker) | — | A generar tras Silver | ⏳ |

**AVISO sobre Red Hat:** `redhat-outputs.zip` (248 MB) **NO contiene issues Jira** — son datos de modelo matemático. Solo usar `redhat-inputs.zip`.

---

## 4. EDAs — Estado y Hallazgos Clave

### EDA_01 — PROMISE ✅ Corregido
- Archivo: `EDA_01_PROMISE_corregido.ipynb` — ejecutar en **Colab**
- Columnas CK disponibles: `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc`
- Target: columna `bug` → binarizar `(bug > 0) = 1`
- Sin nulos en `bug-data/`
- Proyectos: ant, camel, jedit, log4j, lucene, poi

### EDA_02 — Apache Jira ✅ Corregido
- Archivo: `EDA_02_APACHE_JIRA_corregido.ipynb` — ejecutar en **Colab**
- Colecciones útiles: `issues`, `events` (descartar comments, worklogs, users)
- Cycle Time: `resolutiondate - created`
- Extracción proyecto: `key.split('-')[0]` → `HADOOP-1234 → HADOOP`
- Cols a eliminar (>50% nulos): `regression`, `patchinfo`, `environment`, `duedate`, `timeestimate`

### EDA_03 — Red Hat Jira ✅ Ejecutado localmente (505,096 issues validados)
- Archivo: `EDA_03_REDHAT_JIRA.ipynb` — ejecutado localmente (Colab crasheaba por OOM)
- Parquets generados en: `redhat_parquet/` (251 archivos, ~12 MB en disco)
- Schema real: `Issue key | Issue Type | Status | Project key | Created | Resolved`
- **Fechas en formato `DD/MM/YYYY HH:MM` — `dayfirst=True` OBLIGATORIO**

| Métrica | Valor |
|---|---|
| Issues con resolución | 436,475 / 505,096 (86.4%) |
| Cycle Time p50 | **28 días** |
| Cycle Time p75 | 127 días |
| Bug ratio (bugs/tasks+stories) | **1.21** |
| Cols a eliminar | `Fix Version/s`, `Assignee`, `Updated` (>99% nulos) |

### EDA_04 — GHArchive ✅ Ejecutado localmente (24/24 archivos)
- Archivo: `EDA_04_GHARCHIVE.ipynb` — ejecutado localmente
- **Filtro obligatorio**: `repo_name.startswith('apache/')` — sin esto el CFR es basura
- 4,994 eventos Apache cargados de 24 archivos (1.69 GiB procesados)

| Métrica | Valor |
|---|---|
| Change Failure Rate | **26.1%** (era 77.3% sin filtro — incorrecto) |
| Deploy Freq top repos | airflow (10), camel (11), spark (6) |
| Evento dominante | IssueCommentEvent (21.9%) |
| Rango temporal | 2022-01-01 → 2022-03-15 |

---

## 5. Capa Silver ✅ COMPLETA

Todos los scripts están en `gs://shiftmetrics-bronze/scripts/`. Outputs verificados en `gs://shiftmetrics-silver/`.

| Job | Script | Output | Partición |
|-----|--------|--------|-----------|
| 01 PROMISE | `silver_job_01_promise.py` | `gs://shiftmetrics-silver/promise/` | project |
| 02 Apache Jira | `silver_job_02_apache_jira.py` | `gs://shiftmetrics-silver/apache-jira/` | project |
| 03 Red Hat Jira | `silver_job_03_redhat_jira.py` | `gs://shiftmetrics-silver/redhat-jira/` | project_key (251) |
| 04 GHArchive | `silver_job_04_gharchive.py` | `gs://shiftmetrics-silver/gharchive/` | year, month |

**Correcciones aplicadas (importantes para re-runs):**
- Job 01: path real es `bug-data/*/*.csv` (CSVs en subcarpetas por proyecto, no en raíz)
- Job 02: path real es `apache-jira-parquet/issues/` · columna tipo = `issuetype_name` (struct aplanado)
- Job 03: parquets Red Hat subidos a `gs://shiftmetrics-bronze/redhat-jira-parquet/` (Spark no lee ZIPs)
- Job 04: sin correcciones

**Schemas clave:**
- Silver 02: `issuetype_name` (no `issuetype`) — el BSON fue aplanado por `bson_to_parquet.py`
- Silver 04: columna `apache_project_key` = upper(split(repo_name,"/")[1]) → "SPARK", "HADOOP" — clave de join con Jira

---

## 6. Capa Gold — `sprint_features` en BigQuery ✅ COMPLETA

Tabla final: **una fila por sprint**. Construida uniendo las 4 fuentes Silver vía job PySpark en Dataproc.

### Resultados verificados en BigQuery

| Métrica | Valor |
|---|---|
| Total filas | **42,747** |
| Proyectos distintos | 619 |
| Sprints distintos | 250 |
| Defectos positivos | 30,116 |
| Prevalencia `defecto_escapado=1` | **70.5%** |

**Desglose de filas por fuente Silver:**

| Fuente | Filas aportadas |
|---|---|
| Apache Jira (base del JOIN) | 42,747 |
| GHArchive (joined) | 205 |
| Red Hat Jira (joined) | 17,475 |
| PROMISE CK (joined) | 11 proyectos |

### Script ejecutado
`gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py`

### Notas de producción
- El flag `--jars` **no aplica al crear clusters** en Dataproc — aplica al enviar jobs.
  Dataproc incluye el conector BigQuery por defecto, por lo que el job corrió **sin `--jars`**.
- Cluster usado: `shiftmetrics-cluster` con `n1-standard-2` (2 CPUs/nodo × 3 nodos = 6 CPUs),
  dentro de la cuota de 8 CPUs.
- **Cluster eliminado** tras verificar la tabla para conservar créditos.

### Consideraciones para ML
- **Prevalencia alta (70.5%):** Viable, pero se recomienda evaluar:
  - Undersampling de clase mayoritaria, o
  - Ajustar umbral de `cycle_time_days` de 30 → 60 días para reducir positivos
- **NULLs en métricas CK:** `avg_wmc`, `avg_cbo`, etc. son NULL para proyectos fuera de PROMISE.
  Opciones: imputar por mediana, o filtrar a proyectos Apache con datos CK completos.

Tabla destino:

```sql
sprint_id            STRING    -- {project}_{sprint_number}
project              STRING
sprint_number        INT64
-- CK Metrics (PROMISE)
avg_wmc              FLOAT64
avg_dit              FLOAT64
avg_cbo              FLOAT64
avg_rfc              FLOAT64
avg_lcom             FLOAT64
avg_loc              FLOAT64
-- Jira Metrics (Apache + Red Hat)
bug_story_ratio      FLOAT64
avg_cycle_time_days  FLOAT64
num_bugs_sprint      INT64
num_stories_sprint   INT64
-- DORA Metrics (GHArchive)
deploy_frequency     FLOAT64   -- pushes/semana a main/master
change_failure_rate  FLOAT64   -- PRs sin merge / total PRs
-- Target
defecto_escapado     INT64     -- 0 o 1
```

---

## 7. Parámetros para Dataset Sintético (calibrados con EDAs reales)

```python
# Calibración basada en EDA_03 (Red Hat) y EDA_04 (GHArchive)
cycle_time_p50       = 28      # días (Red Hat Jira)
bug_story_ratio_rh   = 1.21    # Red Hat (más bugs que historias)
cfr_proxy            = 0.261   # GHArchive Apache repos
issues_resueltos_pct = 0.864   # 86.4% tienen fecha de resolución
n_sprints            = 2_000

# Distribuciones sugeridas
avg_cycle_time = np.random.lognormal(mean=3.3, sigma=1.2, size=n_sprints)  # p50≈28d
bug_story_ratio = np.random.lognormal(mean=0.18, sigma=0.6, size=n_sprints)
change_failure_rate = np.random.beta(a=2.6, b=7.4, size=n_sprints)        # media≈0.26
```

---

## 8. Hoja de Ruta — Prioridades Actuales

| Prioridad | Tarea | Estado |
|---|---|---|
| 1 | Silver Job 01: PROMISE | ✅ Ejecutado y verificado |
| 2 | Silver Job 02: Apache Jira | ✅ Ejecutado y verificado |
| 3 | Silver Job 03: Red Hat Jira | ✅ Ejecutado y verificado |
| 4 | Silver Job 04: GHArchive | ✅ Ejecutado y verificado |
| 5 | Crear dataset BigQuery `shiftmetrics_gold` | ✅ Ya existía — verificado |
| 6 | Escribir Gold job PySpark (unión de 4 Silver → sprint_features) | ✅ Ejecutado |
| 7 | Ejecutar Gold job en Dataproc con conector BigQuery | ✅ 42,747 filas verificadas |
| **8** | **Decisión prevalencia: undersample o ajustar umbral cycle_time** | **⏳ SIGUIENTE** |
| **9** | **ML Pipeline (LR baseline + XGBoost + SHAP)** | **⏳ SIGUIENTE** |
| 10 | Dataset 5 — Simulador Sintético (SimPy + Faker) | ⏳ Parámetros listos |
| 11 | Deploy Cloud Run (API REST) | ⏳ |

**Nota Gold:** recrear cluster con jar BigQuery:
```bash
gcloud dataproc clusters create shiftmetrics-cluster \
  --region=us-central1 --zone=us-central1-a \
  --master-machine-type=n1-standard-4 --worker-machine-type=n1-standard-4 \
  --num-workers=2 --master-boot-disk-size=50GB --worker-boot-disk-size=50GB \
  --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
  --project=shiftmetrics-analytics
```

---

*Documento actualizado 2026-05-22 · ShiftMetrics Analytics · EAFIT*
