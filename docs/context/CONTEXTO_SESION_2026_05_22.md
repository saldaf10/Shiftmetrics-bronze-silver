# ShiftMetrics — Contexto para Compañero de Equipo
**Sesión: 2026-05-22 · Quién lo hizo: Juan (juanmejia0317@gmail.com)**

Este documento resume exactamente qué se hizo en esta sesión para que puedas continuar desde Gold sin perder contexto. Pásale este archivo a Claude al inicio de tu sesión.

---

## Estado del Pipeline HOY

```
Bronze ✅ completo → Silver ✅ completo → Gold ⏳ SIGUIENTE → ML ⏳ → Deploy ⏳
```

---

## Lo que se completó en esta sesión

### Silver — 4 jobs ejecutados y verificados en Dataproc ✅

Todos los scripts están en `gs://shiftmetrics-bronze/scripts/` y los outputs en `gs://shiftmetrics-silver/`.

| Job | Script | Output verificado |
|-----|--------|-------------------|
| 01 PROMISE | `silver_job_01_promise.py` | `gs://shiftmetrics-silver/promise/project=ant/`, `project=camel/`, ... `project=xerces/` |
| 02 Apache Jira | `silver_job_02_apache_jira.py` | `gs://shiftmetrics-silver/apache-jira/project=HADOOP/`, `project=SPARK/`, ... |
| 03 Red Hat Jira | `silver_job_03_redhat_jira.py` | `gs://shiftmetrics-silver/redhat-jira/project_key=ACM/`, ... 251 particiones |
| 04 GHArchive | `silver_job_04_gharchive.py` | `gs://shiftmetrics-silver/gharchive/year=2022/month=1/`, ... `month=12/` |

**Correcciones aplicadas durante la ejecución:**
- Job 01: path corregido a `bug-data/*/*.csv` (CSVs están en subcarpetas por proyecto)
- Job 02: path corregido a `apache-jira-parquet/issues/` + columna `issuetype_name` (no `issuetype`)
- Job 03: pre-requisito cumplido — 251 parquets subidos a `gs://shiftmetrics-bronze/redhat-jira-parquet/`
- Job 04: sin cambios, corrió limpio

---

## Schemas Silver producidos

### Silver 01 — PROMISE
```
project, version, wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc,
dam, moa, mfa, cam, ic, cbm, amc,
bug_count (float), defect_flag (0/1), defect_density (float), total_modules
Partición: project
```

### Silver 02 — Apache Jira
```
key, project, issuetype_name, issue_category (bug/story/task/other),
status, resolution, created_ts, resolution_ts, cycle_time_days,
sprint (YYYY-M), bug_story_ratio (float)
Partición: project
Nota: columna de tipo se llama issuetype_name (struct aplanado por bson_to_parquet.py)
```

### Silver 03 — Red Hat Jira
```
issue_key, issue_type, issue_category (bug/story/task/other), status,
project_key, project_name, resolution, created_ts, resolved_ts,
cycle_time_days, sprint (YYYY-M),
num_bugs, num_stories, num_tasks, total_issues_sprint, bug_story_ratio (float)
Partición: project_key
```

### Silver 04 — GHArchive
```
repo_name, apache_project_key (ej: "SPARK", "HADOOP"),
year, month,
push_count, deploy_frequency_weekly (push_count / 4.33),
total_prs_closed, prs_merged, prs_not_merged,
change_failure_rate (prs_not_merged / total_prs_closed)
Partición: year, month
```

---

## GCP — Recursos actuales

| Recurso | Nombre |
|---------|--------|
| Proyecto | `shiftmetrics-analytics` (ID: 919593201130) |
| Bucket Bronze | `gs://shiftmetrics-bronze` |
| Bucket Silver | `gs://shiftmetrics-silver` |
| Dataset BigQuery (a crear) | `shiftmetrics_gold` |
| Tabla BigQuery (a crear) | `shiftmetrics_gold.sprint_features` |
| Cluster Dataproc | `shiftmetrics-cluster` — **APAGADO** (recrear para Gold) |
| Región | `us-central1` |

**Permisos ya configurados:**
- Service account `919593201130-compute@developer.gserviceaccount.com` tiene `roles/storage.objectAdmin` en el proyecto
- IAM en buckets de staging/temp de Dataproc ya configurado

**Para recrear el cluster cuando lo necesites:**
```bash
gcloud dataproc clusters create shiftmetrics-cluster \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-4 \
  --num-workers=2 \
  --master-boot-disk-size=50GB \
  --worker-boot-disk-size=50GB \
  --project=shiftmetrics-analytics
```

---

## Lo que sigue: Capa Gold

### Objetivo
Crear la tabla `shiftmetrics_gold.sprint_features` en BigQuery — una fila por `(project, sprint)` uniendo las 4 fuentes Silver.

### Schema objetivo de sprint_features
```sql
sprint_id              STRING    -- "{project}_{sprint}" ej: "SPARK_2022-3"
project                STRING    -- clave del proyecto
sprint                 STRING    -- "YYYY-M" ej: "2022-3"
-- CK Metrics (de Silver PROMISE — promedio por proyecto/versión)
avg_wmc                FLOAT64
avg_dit                FLOAT64
avg_cbo                FLOAT64
avg_rfc                FLOAT64
avg_lcom               FLOAT64
avg_loc                FLOAT64
defect_density         FLOAT64   -- módulos defectuosos / total módulos
-- Jira Metrics (Apache Jira o Red Hat Jira — unir por project)
bug_story_ratio        FLOAT64
avg_cycle_time_days    FLOAT64
num_bugs_sprint        INT64
num_stories_sprint     INT64
-- DORA Metrics (GHArchive — unir por apache_project_key + year/month)
deploy_frequency_weekly FLOAT64  -- pushes/semana a main/master
change_failure_rate    FLOAT64   -- PRs sin merge / total PRs closed
-- Target
defecto_escapado       INT64     -- 0 o 1 (a definir/generar)
```

### Estrategia de unión
```
Silver Apache Jira  (project, sprint)
    LEFT JOIN Silver PROMISE  ON project (CK metrics son por proyecto, no por sprint)
    LEFT JOIN Silver GHArchive ON apache_project_key=project AND year/month=sprint
    LEFT JOIN Silver Red Hat   ON project_key=project AND sprint=sprint
→ BigQuery: shiftmetrics_gold.sprint_features
```

**Clave de join GHArchive:**
- GHArchive tiene columna `apache_project_key` = upper(split(repo_name, "/")[1])
- Ej: "apache/spark" → "SPARK" — coincide con Apache Jira project key

### Pasos para Gold
1. Crear dataset BigQuery: `bq mk --dataset shiftmetrics-analytics:shiftmetrics_gold`
2. Escribir Gold job PySpark que lea los 4 Silver y los una
3. El Gold job escribe a BigQuery usando el conector Spark-BigQuery
4. Recrear cluster Dataproc con el conector BigQuery activado
5. Ejecutar Gold job
6. Verificar tabla en BigQuery Console

### Conector BigQuery para Dataproc
El cluster necesita el jar del conector. Al crear el cluster agregar:
```bash
--jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar
```

---

## Métricas de sanidad para verificar Silver antes de Gold

```bash
# Verificar que los 4 outputs existen
gsutil ls gs://shiftmetrics-silver/promise/
gsutil ls gs://shiftmetrics-silver/apache-jira/
gsutil ls gs://shiftmetrics-silver/redhat-jira/
gsutil ls gs://shiftmetrics-silver/gharchive/
```

Valores esperados:
- PROMISE: 11 proyectos (ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces)
- Apache Jira: proyectos como HADOOP, SPARK, HBASE, HDFS, HIVE, etc.
- Red Hat Jira: 251 project_keys
- GHArchive: year=2022/month=1/ hasta month=12/

---

## Scripts locales (en la máquina de Juan)

```
/home/camus/Desktop/Data_Science/Integrador/
├── Jobs/
│   ├── silver_job_01_promise.py       ✅
│   ├── silver_job_02_apache_jira.py   ✅ (path y columna issuetype_name corregidos)
│   ├── silver_job_03_redhat_jira.py   ✅
│   └── silver_job_04_gharchive.py     ✅
├── Context/
│   ├── ShiftMetrics_Contexto_y_Hoja_de_Ruta.md  (actualizado hoy)
│   └── CONTEXTO_SESION_2026_05_22.md            (este archivo)
├── EDAS/
│   ├── EDA_01_PROMISE_corregido.ipynb
│   ├── EDA_02_APACHE_JIRA_corregido.ipynb
│   ├── EDA_03_REDHAT_JIRA.ipynb       (corregido, correr local)
│   └── EDA_04_GHARCHIVE.ipynb         (corregido)
└── redhat_parquet/                    (251 parquets ya subidos a GCS)
```

---

*ShiftMetrics Analytics · EAFIT SI7006 · 2026-05-22*
