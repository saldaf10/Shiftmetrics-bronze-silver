# ShiftMetrics Analytics — Contexto de Sesión
**Proyecto Integrador · Master Data Science & Analytics · EAFIT**
**Equipo SI7006 · Sesión del: 2026-05-21**

---

## 1. Estado General: Capa Bronze ✅ COMPLETA

Todos los datasets ya están en `gs://shiftmetrics-bronze`. No se requiere ninguna acción adicional en Bronze.

| Dataset | Ruta GCS | Estado |
|---|---|---|
| PROMISE (métricas CK) | `bronze/promise/PROMISE-backup/bug-data/*.csv` | ✅ |
| Apache Jira | `bronze/apache-jira-parquet/` (6 colecciones, Parquet) | ✅ |
| Red Hat Jira | `bronze/redhat-jira/redhat-inputs.zip` | ✅ |
| GHArchive 2022 | `bronze/gharchive/*.json.gz` (24 archivos, 1.69 GiB) | ✅ |

---

## 2. Estado de los EDAs

### EDA_01 — PROMISE (`EDA_01_PROMISE_corregido.ipynb`) ✅ Listo
Notebook corregido y funcional. **Requiere ejecutarse en Google Colab** (usa `from google.colab import auth`).

**Correcciones ya aplicadas:**
- Sintaxis rota: `[m.lower() for m in]` → `[m.lower() for m in CK_METRICS]`
- Carpeta incorrecta: leía `AST_encoding/` → corregido a `bug-data/`
- Variable mal nombrada: `hal_cols` → `ck_cols`
- Carga parcial de 10 CSVs → carga de **todos** los CSVs

**Parámetros clave para Silver y Sintético:**
- Subcarpeta: `promise/PROMISE-backup/bug-data/`
- Columnas CK: `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc`
- Target: columna `bug` → binarizar con `(bug > 0) = 1`
- Proyectos referencia: ant, camel, jedit, log4j, lucene, poi
- Nulos: ninguno detectado en `bug-data/`

---

### EDA_02 — Apache Jira (`EDA_02_APACHE_JIRA_corregido.ipynb`) ✅ Listo
Notebook corregido y funcional. **Requiere ejecutarse en Google Colab.**

**Correcciones ya aplicadas:**
- Clasificador de colecciones roto → regla manual `if coll_name == 'issues'`
- Transiciones buscadas en colección separada → extraídas de `events.items` filtrando `field == 'status'`
- Cargaba solo 1 parquet por colección → ahora carga hasta 5 (`SAMPLE_FILES = 5`)
- Métricas añadidas: Cycle Time directo, Bug-to-Story Ratio, extracción de `project` desde `key`

**Parámetros clave para Silver y Sintético:**
- Colecciones útiles: `issues`, `events` (descartar `comments`, `worklogs`, `users`)
- Cycle Time: `resolutiondate - created` en días
- Extracción de proyecto: `key.split('-')[0]` → `HADOOP-1234 → HADOOP`
- Columnas a eliminar (>50% nulos): `regression`, `patchinfo`, `environment`, `duedate`, `timeestimate`
- Bug-to-Story Ratio: `bugs / (stories + tasks)` por proyecto

---

### EDA_03 — Red Hat Jira (`EDA_03_REDHAT_JIRA.ipynb`) ✅ Ejecutado localmente

**Contexto:** Colab crasheaba con OOM al intentar cargar `redhat-outputs.zip` (248 MB). Se resolvió ejecutando en local con `MODE = 'local'`, que escribe los datos a parquet en disco sin cargar todo en RAM.

**Correcciones aplicadas en esta sesión:**
1. `zip_blobs` no definida → agregado guard `if 'zip_blobs' not in locals()`
2. Descarga parcial del ZIP rota → reemplazado por `tempfile.NamedTemporaryFile` (descarga completa)
3. Solo procesaba algunos CSVs → ahora itera todos los CSVs de ambos ZIPs
4. Fechas mal interpretadas → añadido `dayfirst=True` en todos los `pd.to_datetime()`
5. `MODE='local'` dejaba `dfs=[]` → **nueva celda 3b** que lee los parquets generados y popula `dfs` para que las celdas de análisis ejecuten
6. Celda de issue types no encontraba columna → añadido `'Issue Type'` a `type_candidates`
7. Cycle Time sin `dayfirst=True` → corregido

**Hallazgo crítico — `redhat-outputs.zip` NO es datos Jira:**
El ZIP de 248 MB contiene CSVs con columnas `['Time', 'beta', 'alpha', 'epsilon', 'gamma', 'm']` (datos de modelo matemático). **No usar en el pipeline.** Todo el dataset real está en `redhat-inputs.zip`.

**Resultados validados (505,096 issues, 251 proyectos):**
| Métrica | Valor |
|---|---|
| Total issues | 505,096 |
| Proyectos únicos | 251 |
| Columnas reales | `Issue key`, `Issue Type`, `Status`, `Project key`, `Created`, `Resolved` |
| Formato de fechas | `DD/MM/YYYY HH:MM` — `dayfirst=True` OBLIGATORIO |
| Issues con resolución | 436,475 / 505,096 (86.4%) |
| Cycle Time p50 | **28 días** |
| Cycle Time p75 | 127 días |
| Cycle Time p90 | 400 días |
| Bug ratio | 1.21 (223K bugs / 184K tasks+stories) |
| Cols a eliminar (>50% nulos) | `Fix Version/s`, `Assignee`, `Updated` |

**Cómo ejecutar (local, sin Colab):**
1. Celdas 0–5: ya tienen outputs válidos, no re-ejecutar
2. Celda 6 (carga ZIPs): ya escribió los parquets en `redhat_parquet/` — **no re-ejecutar**
3. **Celda 3b en adelante**: ejecutar secuencialmente — lee parquets y corre el análisis

---

### EDA_04 — GHArchive (`EDA_04_GHARCHIVE.ipynb`) ✅ Ejecutado localmente

**Correcciones aplicadas en esta sesión:**
1. CFR calculado sin filtro Apache → filtro `repo_name.startswith('apache/')` aplicado **en tiempo de carga**
2. Muestra de solo 2 archivos / 10,000 eventos (código viejo) → ahora itera los **24 archivos** completos
3. Filtro `actor_login` demasiado amplio (capturaba users con "apache" en username) → **eliminado**, solo se usa `repo_name`
4. `AttributeError: DatetimeIndex has no .dt` en celda temporal → corregido con `pd.Series(pd.to_datetime(...))`

**Resultados validados (24 archivos procesados):**
| Métrica | Valor |
|---|---|
| Archivos procesados | 24 de 24 (1.69 GiB total) |
| Eventos Apache cargados | 4,994 |
| Repos más activos | airflow, spark, arrow, pulsar, beam |
| Deployment Frequency top | `apache/camel` (11), `apache/airflow` (10) |
| Change Failure Rate | **26.1%** (PRs cerrados sin merge / total PRs cerrados) |
| Rango temporal | 2022-01-01 → 2022-03-15 |
| Evento dominante | `IssueCommentEvent` (21.9%) |

**Nota:** El CFR anterior (77.3%) era basura — calculado sobre todos los repos del mundo sin filtro Apache.

---

## 3. Pasos a Seguir — Capa Silver (próxima prioridad)

La Bronze está completa. El siguiente paso es construir **4 jobs PySpark en Dataproc** que limpian y transforman los datos a Parquet en `gs://shiftmetrics-silver/`.

### Silver Job 01 — PROMISE ⏳ Pendiente escribir
```
Input:  gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*.csv
Output: gs://shiftmetrics-silver/promise/
Transformaciones:
  - Estandarizar nombres de columnas a lowercase
  - Castear métricas CK a float64
  - Eliminar filas con bug = null
  - Binarizar bug: (bug > 0).cast(IntegerType) → defect_flag
  - Extraer project y version del nombre del archivo
  - Particionar por: project
```

### Silver Job 02 — Apache Jira ✅ Script creado (`silver_job_02_apache_jira.py`)
```
Input:  gs://shiftmetrics-bronze/apache-jira-parquet/issues*.parquet
Output: gs://shiftmetrics-silver/apache-jira/
Comando Dataproc:
  gcloud dataproc jobs submit pyspark \
    gs://shiftmetrics-bronze/scripts/silver_job_02_apache_jira.py \
    --cluster=shiftmetrics-cluster --region=us-central1 \
    --project=shiftmetrics-analytics
```

### Silver Job 03 — Red Hat Jira ⏳ Pendiente escribir
```
Input:  redhat_parquet/redhat-inputs_*.parquet  (ya generados localmente)
        O directamente gs://shiftmetrics-bronze/redhat-jira/redhat-inputs.zip
Output: gs://shiftmetrics-silver/redhat-jira/
Transformaciones críticas (basadas en EDA_03):
  - IGNORAR redhat-outputs.zip (no es datos Jira)
  - Parsear fechas con dayfirst=True (formato DD/MM/YYYY HH:MM)
  - Calcular cycle_time_days = Resolved - Created
  - Clasificar Issue Type: bug | story | task | other
  - Calcular bug_story_ratio por Project key
  - Eliminar columnas: Fix Version/s, Assignee, Updated (>50% nulos)
  - Particionar por: Project key
```

### Silver Job 04 — GHArchive ⏳ Pendiente escribir
```
Input:  gs://shiftmetrics-bronze/gharchive/*.json.gz
Output: gs://shiftmetrics-silver/gharchive/
Transformaciones críticas (basadas en EDA_04):
  - Filtro: repo_name.startswith('apache/') — OBLIGATORIO
  - Separar PushEvent y PullRequestEvent
  - Deployment Frequency = pushes a refs/heads/main o refs/heads/master por repo/mes
  - Change Failure Rate = PRs closed sin merge / total PRs closed por repo/mes
  - Aplanar campos: repo.name, actor.login, payload.ref, created_at
  - Particionar por: año, mes
```

---

## 4. Después de Silver — Capa Gold y ML

```
Fase 3: Gold → tabla BigQuery sprint_features (una fila por sprint)
         JOIN: Apache Jira + Red Hat Jira + GHArchive + PROMISE
         Comando: bq mk --table shiftmetrics-analytics:gold.sprint_features schema.json

Fase 4: Dataset Sintético (SimPy + Faker)
         Calibrado con parámetros reales de los EDAs:
           cycle_time_p50 = 28 días (Red Hat) / ver Apache
           bug_story_ratio ~ 1.21 (Red Hat) / 0.23 (Apache — validar al correr EDA_02)
           cfr_proxy = 0.261 (GHArchive)
           n_sprints = 2,000

Fase 5: ML Pipeline
         Split temporal 70/15/15
         Modelo 1: Logistic Regression (baseline)
         Modelo 2: XGBoost + SHAP
         Métrica: F2-Score

Fase 6: Deploy → Cloud Run (API REST)
```

---

## 5. Archivos Actuales en el Proyecto

```
/home/camus/Desktop/Data_Science/Integrador/
├── EDA_01_PROMISE_corregido.ipynb          ← ✅ Corregido (ejecutar en Colab)
├── EDA_02_APACHE_JIRA_corregido.ipynb      ← ✅ Corregido (ejecutar en Colab)
├── EDA_03_REDHAT_JIRA.ipynb                ← ✅ Ejecutado localmente, outputs válidos
├── EDA_04_GHARCHIVE.ipynb                  ← ✅ Ejecutado localmente, outputs válidos
├── silver_job_02_apache_jira.py            ← ✅ Script PySpark listo para Dataproc
├── redhat_parquet/                         ← ✅ 251 parquets de redhat-inputs (505K issues)
├── EDA_03_OUTPUTS.md                       ← Métricas validadas EDA_03
├── ShiftMetrics_Contexto_y_Hoja_de_Ruta.md
└── SLT_final.pdf
```

---

*Documento generado en sesión del 2026-05-21 · ShiftMetrics Analytics · EAFIT*
