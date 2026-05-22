# ShiftMetrics — Sesión Gold Completada
**Sesión: 2026-05-22 · Capa Gold ejecutada y verificada**

Este documento es la bitácora de la sesión en que se completó la capa Gold del pipeline.
Pásaselo a Claude al inicio de la próxima sesión (ML Pipeline).

---

## Estado del Pipeline al cierre de esta sesión

```
Bronze ✅  →  Silver ✅  →  Gold ✅  →  ML ⏳ SIGUIENTE  →  Deploy ⏳
```

---

## Lo que se completó en esta sesión

### Paso 1 — Silver verificada ✅
Las 4 tablas Silver están presentes con datos en `gs://shiftmetrics-silver/`:

| Ruta | Estado |
|---|---|
| `gs://shiftmetrics-silver/apache-jira/` | ✅ Con datos |
| `gs://shiftmetrics-silver/redhat-jira/` | ✅ Con datos |
| `gs://shiftmetrics-silver/gharchive/` | ✅ Con datos |
| `gs://shiftmetrics-silver/promise/` | ✅ Con datos |

### Paso 2 — Dataset BigQuery confirmado ✅
El dataset `shiftmetrics_gold` ya existía en el proyecto — no fue necesario crearlo.

### Paso 3 — Cluster Dataproc recreado ✅
Se recreó `shiftmetrics-cluster` con configuración ajustada a la cuota disponible:

```bash
gcloud dataproc clusters create shiftmetrics-cluster \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n1-standard-2 \
  --worker-machine-type=n1-standard-2 \
  --num-workers=2 \
  --project=shiftmetrics-analytics
```

> **Nota:** Se usó `n1-standard-2` (2 CPUs/nodo) en lugar de `n1-standard-4` para respetar la cuota de 8 CPUs del proyecto (1 master + 2 workers = 3 nodos × 2 CPUs = 6 CPUs totales).

> **Corrección de documentación anterior:** El flag `--jars` **no aplica al crear clusters**, sino al **enviar jobs**. Dataproc incluye el conector BigQuery por defecto en versiones recientes, por lo que el job Gold corrió **sin necesidad de especificar `--jars`**.

### Paso 4 — Script subido a GCS ✅
```bash
gsutil cp gold_job_sprint_features.py \
  gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py
```

### Paso 5 — Gold job ejecutado exitosamente ✅
```bash
gcloud dataproc jobs submit pyspark \
  gs://shiftmetrics-bronze/scripts/gold_job_sprint_features.py \
  --cluster=shiftmetrics-cluster \
  --region=us-central1 \
  --project=shiftmetrics-analytics
```

**Filas aportadas por cada fuente Silver:**

| Fuente | Filas |
|---|---|
| Apache Jira (base JOIN) | 42,747 |
| Red Hat Jira (joined) | 17,475 |
| GHArchive (joined) | 205 |
| PROMISE CK (joined) | 11 proyectos |

### Paso 6 — Tabla verificada en BigQuery ✅

```sql
-- Query de verificación ejecutada
SELECT
  COUNT(*)                                    AS total_filas,
  COUNT(DISTINCT project)                     AS proyectos,
  COUNT(DISTINCT sprint)                      AS sprints,
  SUM(defecto_escapado)                       AS defectos_positivos,
  ROUND(AVG(defecto_escapado) * 100, 1)       AS prevalencia_pct
FROM shiftmetrics_gold.sprint_features;
```

**Resultado:**

| total_filas | proyectos | sprints | defectos_positivos | prevalencia_pct |
|---|---|---|---|---|
| 42,747 | 619 | 250 | 30,116 | 70.5% |

### Paso 7 — Cluster eliminado ✅
```bash
gcloud dataproc clusters delete shiftmetrics-cluster \
  --region=us-central1 --project=shiftmetrics-analytics
```
Cluster apagado para conservar los créditos GCP (~$275 disponibles).

---

## Tabla resultante: sprint_features

**Ruta BigQuery:** `shiftmetrics-analytics:shiftmetrics_gold.sprint_features`

**Schema:**
```
sprint_id               STRING     -- "{project}_{sprint}" ej: "SPARK_2022-3"
project                 STRING
sprint                  STRING     -- "YYYY-M"
avg_wmc                 FLOAT64    -- NULL para proyectos fuera de PROMISE
avg_dit                 FLOAT64    -- NULL para proyectos fuera de PROMISE
avg_cbo                 FLOAT64    -- NULL para proyectos fuera de PROMISE
avg_rfc                 FLOAT64    -- NULL para proyectos fuera de PROMISE
avg_lcom                FLOAT64    -- NULL para proyectos fuera de PROMISE
avg_loc                 FLOAT64    -- NULL para proyectos fuera de PROMISE
defect_density          FLOAT64    -- NULL para proyectos fuera de PROMISE
bug_story_ratio         FLOAT64
avg_cycle_time_days     FLOAT64
num_bugs_sprint         INT64
num_stories_sprint      INT64
deploy_frequency_weekly FLOAT64    -- NULL si proyecto no está en GHArchive Apache
change_failure_rate     FLOAT64    -- NULL si proyecto no está en GHArchive Apache
defecto_escapado        INT64      -- 0 o 1 (target ML)
```

---

## Consideraciones críticas para el ML Pipeline

### 1. Prevalencia alta: 70.5%

La clase positiva (`defecto_escapado=1`) representa el 70.5% del dataset. Esto es usable pero puede sesgar el modelo hacia predecir siempre "defecto". Opciones a evaluar:

**Opción A — Undersampling de clase mayoritaria**
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=0.6)  # ratio 60/40 o 50/50
X_res, y_res = rus.fit_resample(X, y)
```

**Opción B — Ajustar umbral de cycle_time_days**
Si el target `defecto_escapado` se construye como `cycle_time_days > threshold_days`:
```python
# Cambiar threshold de 30 días → 60 días reduce la prevalencia
threshold_days = 60   # en vez de 30
df['defecto_escapado'] = (df['avg_cycle_time_days'] > threshold_days).astype(int)
```
Verificar cuánto baja la prevalencia antes de elegir.

**Opción C — `scale_pos_weight` en XGBoost (sin modificar datos)**
```python
# Ratio negatives / positives
scale_pos_weight = (42747 - 30116) / 30116  # ≈ 0.42
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
```

**Opción D — No modificar (F2-Score ya penaliza FN)**
El F2-Score ya está diseñado para priorizar recall. Probar primero sin modificar y medir.

**Recomendación:** Probar primero Opción D (sin cambios), luego Opción A si F2 < 0.6.

---

### 2. NULLs en métricas CK (avg_wmc, avg_cbo, avg_rfc, avg_lcom, avg_loc, defect_density)

Las columnas CK son NULL para proyectos fuera de PROMISE (la mayoría de los 619 proyectos). Estrategias:

| Estrategia | Pros | Contras |
|---|---|---|
| Imputar por mediana | Conserva todos los datos | Introduce ruido artificialmente |
| Filtrar solo proyectos con CK | Datos completos para el modelo | Reduce dataset a ~11 proyectos |
| Excluir columnas CK del modelo | Dataset completo, sin imputar | Pierde información de calidad de código |
| Indicador de NULL como feature | `has_ck_data = 1/0` + imputar | Deja al modelo decidir la importancia |

**Recomendación:** Usar indicador binario `has_ck_data` + imputar CK por mediana. Así el modelo puede aprender que "no tener datos CK" es en sí mismo informativo.

---

### 3. Split temporal

El dataset tiene `sprint = "YYYY-M"` — usar split temporal estricto:
```python
# Ordenar por sprint y hacer corte temporal
train = df[df['sprint'] < '2022-8']    # 70%
val   = df[(df['sprint'] >= '2022-8') & (df['sprint'] < '2022-11')]  # 15%
test  = df[df['sprint'] >= '2022-11']  # 15%
```
**No usar train_test_split aleatorio** — causaría data leakage (el modelo vería datos futuros).

---

## Lo que sigue: ML Pipeline

### Pasos sugeridos para la próxima sesión

1. **Cargar `sprint_features` desde BigQuery:**
   ```python
   from google.cloud import bigquery
   client = bigquery.Client(project='shiftmetrics-analytics')
   df = client.query("SELECT * FROM shiftmetrics_gold.sprint_features").to_dataframe()
   ```

2. **Feature engineering:**
   - Crear columna `has_ck_data` (1 si avg_wmc no es NULL)
   - Imputar NULLs en CK por mediana
   - Imputar NULLs en DORA (deploy_frequency, CFR) por mediana o 0
   - Parsear `sprint` → `year`, `month` como features numéricos

3. **Split temporal 70/15/15**

4. **Baseline Logistic Regression:**
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler
   ```

5. **XGBoost:**
   ```python
   from xgboost import XGBClassifier
   model = XGBClassifier(
       n_estimators=200, max_depth=6, learning_rate=0.1,
       scale_pos_weight=0.42,   # ajustar según prevalencia
       eval_metric='aucpr',
       random_state=42
   )
   ```

6. **Evaluación con F2-Score:**
   ```python
   from sklearn.metrics import fbeta_score
   f2 = fbeta_score(y_test, y_pred, beta=2)
   ```

7. **SHAP values para XGBoost:**
   ```python
   import shap
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test)
   ```

---

## GCP — Estado de recursos al cierre de sesión

| Recurso | Estado |
|---|---|
| `gs://shiftmetrics-bronze` | ✅ Activo — no modificar |
| `gs://shiftmetrics-silver` | ✅ Activo — no modificar |
| `shiftmetrics_gold.sprint_features` | ✅ Tabla lista con 42,747 filas |
| Cluster Dataproc `shiftmetrics-cluster` | **ELIMINADO** — recrear para ML si se usa Dataproc |
| Créditos GCP | ~$275 USD (estimado) |

---

*ShiftMetrics Analytics · EAFIT SI7006 · Sesión Gold completada 2026-05-22*
