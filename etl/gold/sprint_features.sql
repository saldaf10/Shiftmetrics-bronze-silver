-- =============================================================================
-- gold_sprint_features.sql  ·  ShiftMetrics Analytics  ·  EAFIT 2026
-- Capa Gold: tabla sprint_features para ML (SI7009)
--
-- Estrategia de joins (corregida):
--   Spine  : Apache Jira  (619 proyectos, sprints 2000-2021) — backbone
--   Fix #1 : PROMISE CK  → join por UPPER(project) — 4 matches: CAMEL, LUCENE, IVY, SYNAPSE
--   Fix #2 : GHArchive   → agregar a nivel de PROYECTO (sin year/month)
--              Rationale: GHArchive solo tiene 2022; las métricas DORA son
--              características estables del proyecto (se usan como prior)
--   Dropped: Red Hat Jira → sin overlap de project_key con Apache Jira
--
-- Particionamiento: RANGE_BUCKET(sprint_year, 2000–2022) + CLUSTER BY project
-- Conforme con SI7006 requisito de tabla particionada + clusterizada
-- =============================================================================

CREATE OR REPLACE TABLE `shiftmetrics-analytics.shiftmetrics_gold.sprint_features`
PARTITION BY RANGE_BUCKET(sprint_year, GENERATE_ARRAY(2000, 2023, 1))
CLUSTER BY project
AS

-- ============================================================
-- CTE 1: Apache Jira — spine a nivel de (project, sprint)
-- Recalculamos desde Silver para consistencia total
-- ============================================================
WITH apache_spine AS (
  SELECT
    project,
    sprint,
    SAFE_CAST(SPLIT(sprint, '-')[SAFE_OFFSET(0)] AS INT64)  AS sprint_year,
    SAFE_CAST(SPLIT(sprint, '-')[SAFE_OFFSET(1)] AS INT64)  AS sprint_month,
    COUNTIF(issue_category = 'bug')                          AS num_bugs_sprint,
    COUNTIF(issue_category = 'story')                        AS num_stories_sprint,
    COUNTIF(issue_category = 'task')                         AS num_tasks_sprint,
    COUNT(*)                                                  AS total_issues_sprint,
    SAFE_DIVIDE(
      COUNTIF(issue_category = 'bug'),
      NULLIF(COUNTIF(issue_category IN ('story', 'task')), 0)
    )                                                         AS bug_story_ratio,
    AVG(
      CASE
        WHEN cycle_time_days IS NOT NULL AND cycle_time_days >= 0
        THEN CAST(cycle_time_days AS FLOAT64)
      END
    )                                                         AS avg_cycle_time_days
  FROM `shiftmetrics-analytics.shiftmetrics_gold.ext_silver_apache_jira`
  WHERE project IS NOT NULL
    AND sprint IS NOT NULL
  GROUP BY project, sprint
),

-- ============================================================
-- CTE 2: Target — defecto_escapado
-- Un sprint tiene defecto escapado si contiene ≥1 bug que tardó
-- >30 días en resolverse O que aún está abierto (sin resolver)
-- Definición operativa registrada en Model Card (SI7009)
-- ============================================================
target_escape AS (
  SELECT
    project,
    sprint,
    IF(
      COUNTIF(
        issue_category = 'bug'
        AND (cycle_time_days > 30 OR cycle_time_days IS NULL)
      ) > 0,
      1, 0
    ) AS defecto_escapado
  FROM `shiftmetrics-analytics.shiftmetrics_gold.ext_silver_apache_jira`
  WHERE project IS NOT NULL
    AND sprint IS NOT NULL
  GROUP BY project, sprint
),

-- ============================================================
-- CTE 3: PROMISE — CK metrics a nivel de proyecto
-- Fix #1: UPPER(project) para match con Apache Jira keys
-- Proyectos que matchean: CAMEL, LUCENE, IVY, SYNAPSE
-- ============================================================
promise_ck AS (
  SELECT
    UPPER(project)       AS project,
    AVG(wmc)             AS avg_wmc,
    AVG(dit)             AS avg_dit,
    AVG(cbo)             AS avg_cbo,
    AVG(rfc)             AS avg_rfc,
    AVG(lcom)            AS avg_lcom,
    AVG(loc)             AS avg_loc,
    AVG(defect_density)  AS defect_density
  FROM `shiftmetrics-analytics.shiftmetrics_gold.ext_silver_promise`
  WHERE project IS NOT NULL
  GROUP BY UPPER(project)
),

-- ============================================================
-- CTE 4: GHArchive — DORA metrics a nivel de PROYECTO (sin sprint)
-- Fix #2: agregar todo 2022 por proyecto para eliminar restricción temporal
-- Lógica: DORA metrics = características de ingeniería del proyecto (estables)
-- Proyectos con datos: SPARK, HADOOP, HBASE, HIVE, CAMEL, KAFKA, AIRFLOW...
-- ============================================================
dora_project AS (
  SELECT
    apache_project_key                   AS project,
    AVG(deploy_frequency_weekly)         AS deploy_frequency_weekly,
    AVG(change_failure_rate)             AS change_failure_rate,
    SUM(CAST(push_count AS INT64))       AS total_pushes_2022,
    SUM(CAST(total_prs_closed AS INT64)) AS total_prs_2022
  FROM `shiftmetrics-analytics.shiftmetrics_gold.ext_silver_gharchive`
  WHERE apache_project_key IS NOT NULL
  GROUP BY apache_project_key
)

-- ============================================================
-- JOIN FINAL
-- ============================================================
SELECT
  CONCAT(s.project, '_', s.sprint)   AS sprint_id,
  s.project,
  s.sprint,
  COALESCE(s.sprint_year,  0)        AS sprint_year,
  COALESCE(s.sprint_month, 0)        AS sprint_month,

  -- CK Metrics — de PROMISE (solo 4 proyectos; resto = NULL → imputar en ML)
  p.avg_wmc,
  p.avg_dit,
  p.avg_cbo,
  p.avg_rfc,
  p.avg_lcom,
  p.avg_loc,
  p.defect_density,

  -- Jira Metrics — de Apache Jira (619 proyectos, full coverage)
  s.bug_story_ratio,
  s.avg_cycle_time_days,
  CAST(s.num_bugs_sprint    AS INT64)  AS num_bugs_sprint,
  CAST(s.num_stories_sprint AS INT64)  AS num_stories_sprint,
  CAST(s.num_tasks_sprint   AS INT64)  AS num_tasks_sprint,
  CAST(s.total_issues_sprint AS INT64) AS total_issues_sprint,

  -- DORA Metrics — de GHArchive 2022 (proxy de cultura de ingeniería del proyecto)
  g.deploy_frequency_weekly,
  g.change_failure_rate,
  CAST(COALESCE(g.total_pushes_2022, 0) AS INT64) AS total_pushes_2022,
  CAST(COALESCE(g.total_prs_2022,   0) AS INT64) AS total_prs_2022,

  -- Target binario (SI7009)
  COALESCE(t.defecto_escapado, 0)     AS defecto_escapado

FROM apache_spine s
LEFT JOIN target_escape  t  USING (project, sprint)
LEFT JOIN promise_ck     p  ON s.project = p.project
LEFT JOIN dora_project   g  ON s.project = g.project

WHERE s.sprint_year IS NOT NULL
  AND s.sprint_year BETWEEN 2000 AND 2022
;
