# =============================================================================
# Silver Job 03 — Red Hat Jira
# ShiftMetrics Analytics · EAFIT 2026
#
# Input:  gs://shiftmetrics-bronze/redhat-jira-parquet/redhat-inputs_*.parquet
# Output: gs://shiftmetrics-silver/redhat-jira/  (Parquet particionado por project_key)
#
# PRE-REQUISITO — subir los parquets locales a GCS antes de ejecutar este job:
#   gsutil -m cp redhat_parquet/redhat-inputs_*.parquet \
#     gs://shiftmetrics-bronze/redhat-jira-parquet/
#
# Por qué no leer directo del ZIP en Bronze:
#   Spark no puede leer ZIPs de GCS nativamente. Los parquets ya fueron
#   generados localmente en EDA_03 (MODE='local') con el CSV completo.
#   Subirlos a GCS es más simple y barato que descomprimir en Dataproc.
#
# HALLAZGO IMPORTANTE (EDA_03):
#   redhat-outputs.zip NO contiene issues Jira — son datos de modelo matemático.
#   Solo se usa redhat-inputs.zip → redhat_parquet/redhat-inputs_*.parquet
#
# Qué hace este job:
#   1. Lee los 251 parquets de redhat-inputs desde Bronze
#   2. Renombra columnas con espacios a snake_case
#   3. Elimina columnas con >50% nulos (Fix Version/s, Assignee, Updated)
#   4. Filtra filas sin issue_key o sin created_raw
#   5. Parsea fechas en formato DD/MM/YYYY HH:MM (dayfirst — OBLIGATORIO)
#   6. Calcula cycle_time_days = resolved - created
#   7. Clasifica issue_type: bug | story | task | other
#   8. Crea sprint aproximado: año-mes de created
#   9. Calcula bug_story_ratio por project_key + sprint
#  10. Escribe Parquet particionado por project_key en Silver
#
# Cómo ejecutar en Dataproc:
#   gcloud dataproc jobs submit pyspark \
#     gs://shiftmetrics-bronze/scripts/silver_job_03_redhat_jira.py \
#     --cluster=shiftmetrics-cluster \
#     --region=us-central1 \
#     --project=shiftmetrics-analytics
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, datediff, when, lower,
    year, month, concat_ws,
    count, sum as _sum, round as spark_round,
    lit
)
from pyspark.sql.types import IntegerType, DoubleType

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Formato de fecha confirmado en EDA_03: "22/03/2023 17:04"
# En Java/Spark: dd = día (01-31), MM = mes (01-12), yyyy = año, HH:mm = hora
# CRÍTICO: sin este formato los días y meses se invierten (02/11 → noviembre 2, no febrero 11)
DATE_FORMAT = "dd/MM/yyyy HH:mm"

# Mapa de renombrado: columnas originales con espacios → snake_case
# Esto evita usar backticks en todo el job y alinea el schema con Apache Jira Silver
RENAME_MAP = {
    "Issue key":    "issue_key",
    "Issue Type":   "issue_type",
    "Status":       "status",
    "Project key":  "project_key",
    "Project name": "project_name",
    "Project type": "project_type",
    "Resolution":   "resolution",
    "Created":      "created_raw",    # sufijo _raw = string sin parsear aún
    "Resolved":     "resolved_raw",
    "Assignee":     "assignee",       # se eliminará (>99% nulos)
    "Updated":      "updated_raw",    # se eliminará (>99% nulos)
    "Fix Version/s":"fix_versions",   # se eliminará (>99% nulos)
}

# Columnas a eliminar tras el rename (>50% nulos según EDA_03)
COLS_TO_DROP = ["assignee", "updated_raw", "fix_versions"]


def main():
    # -----------------------------------------------------------------------
    # 0. Crear SparkSession
    # -----------------------------------------------------------------------
    spark = SparkSession.builder \
        .appName("ShiftMetrics_Silver_RedHat_Jira") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    input_path  = "gs://shiftmetrics-bronze/redhat-jira-parquet/redhat-inputs_*.parquet"
    output_path = "gs://shiftmetrics-silver/redhat-jira/"

    print("=" * 60)
    print("Silver Job 03 — Red Hat Jira")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Leer los 251 parquets desde Bronze
    # -----------------------------------------------------------------------
    # mergeSchema=true: distintos proyectos pueden tener columnas ligeramente
    # diferentes (e.g., algunos CSVs tienen "Assignee", otros no).
    # Con mergeSchema, Spark unifica el schema y rellena con null donde falta.
    df = spark.read \
        .option("mergeSchema", "true") \
        .parquet(input_path)

    print(f"\nSchema leído desde Bronze:")
    df.printSchema()
    print(f"Filas totales: {df.count():,}")

    # -----------------------------------------------------------------------
    # 2. Renombrar columnas con espacios a snake_case
    # -----------------------------------------------------------------------
    # Los CSVs de Red Hat tienen columnas como "Issue key", "Issue Type", etc.
    # Los espacios y el "/" en "Fix Version/s" complican referencias posteriores.
    # Renombramos todo de una vez al inicio para que el resto del job sea limpio.
    for original, renamed in RENAME_MAP.items():
        if original in df.columns:
            df = df.withColumnRenamed(original, renamed)

    # Eliminar columnas auxiliares que pudo haber agregado el EDA local
    for aux_col in ["_source_file", "_zip_source"]:
        if aux_col in df.columns:
            df = df.drop(aux_col)

    # -----------------------------------------------------------------------
    # 3. Eliminar columnas con más del 50% de nulos
    # -----------------------------------------------------------------------
    # EDA_03 confirmó:
    #   Fix Version/s → 99.96% nulos
    #   Assignee      → 99.95% nulos
    #   Updated       → 99.91% nulos
    # Estas columnas no aportan información útil al modelo y desperdician espacio.
    existing_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if existing_to_drop:
        df = df.drop(*existing_to_drop)
        print(f"\nColumnas eliminadas (>50% nulos): {existing_to_drop}")

    # -----------------------------------------------------------------------
    # 4. Filtrar filas sin issue_key o sin fecha de creación
    # -----------------------------------------------------------------------
    # Un issue sin key no se puede identificar.
    # Un issue sin created_raw no tiene dimensión temporal → inútil para ML.
    before = df.count()
    df = df.filter(
        col("issue_key").isNotNull() &
        col("created_raw").isNotNull()
    )
    after = df.count()
    print(f"Filas eliminadas por issue_key/created nulos: {before - after:,} "
          f"(quedan {after:,})")

    # -----------------------------------------------------------------------
    # 5. Parsear fechas con formato DD/MM/YYYY HH:MM
    # -----------------------------------------------------------------------
    # EDA_03 confirmó que Red Hat usa formato europeo: "22/03/2023 17:04"
    # to_timestamp(columna, formato) convierte el string al tipo Timestamp de Spark.
    #
    # Si el string no matchea el formato → devuelve null (no falla el job).
    # El EDA mostró que sin este formato correcto el p50 de cycle time era 1090 días
    # (absurdo). Con el formato correcto: p50 = 28 días (coherente).
    df = df.withColumn("created_ts",  to_timestamp(col("created_raw"),  DATE_FORMAT)) \
           .withColumn("resolved_ts", to_timestamp(col("resolved_raw"), DATE_FORMAT))

    # Verificar cuántas fechas quedaron nulas tras el parseo
    null_created  = df.filter(col("created_ts").isNull()).count()
    null_resolved = df.filter(col("resolved_ts").isNull()).count()
    total = df.count()
    print(f"\nFechas nulas tras parseo:")
    print(f"  created_ts:  {null_created:,}  ({null_created/total*100:.2f}%)")
    print(f"  resolved_ts: {null_resolved:,} ({null_resolved/total*100:.2f}%) "
          f"— esperado ~13.6% según EDA_03")

    # -----------------------------------------------------------------------
    # 6. Calcular Cycle Time en días
    # -----------------------------------------------------------------------
    # cycle_time_days = resolved_ts - created_ts (en días enteros)
    # datediff(fecha_fin, fecha_inicio) → entero (puede ser negativo si los datos
    # tienen errores). Filtramos cycle_time < 0 más adelante en el job Gold.
    #
    # Un issue sin resolved_ts (aún abierto) tendrá cycle_time_days = null.
    # Esto es correcto: el Gold usará solo issues resueltos para calcular promedios.
    df = df.withColumn(
        "cycle_time_days",
        datediff(col("resolved_ts"), col("created_ts"))
    )

    # -----------------------------------------------------------------------
    # 7. Clasificar issue_type en 4 categorías canónicas
    # -----------------------------------------------------------------------
    # Red Hat tiene tipos como: Bug, Story, Task, Feature Request, Enhancement,
    # Epic, Component Upgrade, Feature, Spike, Release.
    # Los agrupamos en 4 para alinear el schema con Apache Jira Silver.
    #
    # Reglas (en orden de prioridad):
    #   "Bug"                         → bug
    #   "Story", "Epic", "Feature",
    #   "Feature Request"             → story   (trabajo de valor de negocio)
    #   "Task", "Sub-task", "Spike",
    #   "Component Upgrade"           → task    (trabajo técnico)
    #   Todo lo demás                 → other
    df = df.withColumn(
        "issue_category",
        when(lower(col("issue_type")) == "bug", lit("bug"))
        .when(lower(col("issue_type")).isin(
            "story", "epic", "feature", "feature request", "enhancement"
        ), lit("story"))
        .when(lower(col("issue_type")).isin(
            "task", "sub-task", "subtask", "spike",
            "component upgrade", "release"
        ), lit("task"))
        .otherwise(lit("other"))
    )

    # -----------------------------------------------------------------------
    # 8. Crear sprint aproximado: año-mes de created_ts
    # -----------------------------------------------------------------------
    # Red Hat Jira no tiene columna nativa de sprint (igual que Apache Jira).
    # Aproximamos sprint como "YYYY-M" del mes de creación del issue.
    # Esto permite calcular métricas por ventana temporal de ~1 mes.
    #
    # Ejemplo: created_ts = 2023-03-22 → sprint = "2023-3"
    df = df.withColumn(
        "sprint",
        concat_ws("-", year(col("created_ts")), month(col("created_ts")))
    )

    # -----------------------------------------------------------------------
    # 9. Calcular Bug-to-Story Ratio por project_key + sprint
    # -----------------------------------------------------------------------
    # bug_story_ratio = num_bugs / (num_stories + num_tasks) por sprint
    #
    # EDA_03 mostró ratio global de 1.21 — Red Hat tiene más bugs que historias.
    # El Gold usará este ratio como feature por sprint para predecir defecto_escapado.
    #
    # Estrategia:
    #   a) Agregar por (project_key, sprint) → conteos por categoría
    #   b) Calcular ratio
    #   c) Join de vuelta al DataFrame principal (cada issue queda con su ratio de sprint)
    sprint_agg = df.groupBy("project_key", "sprint").agg(
        count("*").alias("total_issues_sprint"),
        _sum(when(col("issue_category") == "bug",   lit(1)).otherwise(lit(0))).alias("num_bugs"),
        _sum(when(col("issue_category") == "story", lit(1)).otherwise(lit(0))).alias("num_stories"),
        _sum(when(col("issue_category") == "task",  lit(1)).otherwise(lit(0))).alias("num_tasks"),
    )

    sprint_agg = sprint_agg.withColumn(
        "num_stories_tasks",
        col("num_stories") + col("num_tasks")
    ).withColumn(
        "bug_story_ratio",
        when(
            col("num_stories_tasks") > 0,
            spark_round(
                col("num_bugs").cast(DoubleType()) / col("num_stories_tasks"),
                3
            )
        ).otherwise(lit(None).cast(DoubleType()))
    )

    print("\nEjemplo de bug_story_ratio por sprint (primeros 10):")
    sprint_agg.orderBy("project_key", "sprint").show(10, truncate=False)

    # Join al DataFrame principal
    # Seleccionamos solo las columnas del agg que necesita el Gold
    df = df.join(
        sprint_agg.select(
            "project_key", "sprint",
            "num_bugs", "num_stories", "num_tasks",
            "total_issues_sprint", "bug_story_ratio"
        ),
        on=["project_key", "sprint"],
        how="left"
    )

    # -----------------------------------------------------------------------
    # 10. Eliminar columnas de string crudas (ya tenemos los timestamps)
    # -----------------------------------------------------------------------
    # created_raw y resolved_raw ya fueron parseadas a created_ts / resolved_ts.
    # Los strings originales no aportan nada adicional en Silver.
    df = df.drop("created_raw", "resolved_raw")

    # -----------------------------------------------------------------------
    # 11. Schema final antes de escribir
    # -----------------------------------------------------------------------
    print("\nSchema final a escribir en Silver:")
    df.printSchema()

    # Estadísticas de cycle_time como sanidad check
    print("\nEstadísticas de cycle_time_days (sanity check — p50 esperado ~28 días):")
    df.filter(col("cycle_time_days").isNotNull() & (col("cycle_time_days") >= 0)) \
      .select("cycle_time_days") \
      .summary("count", "50%", "75%", "90%", "mean", "max") \
      .show()

    # -----------------------------------------------------------------------
    # 12. Escribir en Silver particionado por project_key
    # -----------------------------------------------------------------------
    # Resultado en GCS:
    #   gs://shiftmetrics-silver/redhat-jira/project_key=ACM/part-*.parquet
    #   gs://shiftmetrics-silver/redhat-jira/project_key=ENTESB/part-*.parquet
    #   ...  (251 particiones, una por proyecto Red Hat)
    #
    # El job Gold leerá solo los project_key que se crucen con Apache Jira.
    df.write \
        .mode("overwrite") \
        .partitionBy("project_key") \
        .parquet(output_path)

    print(f"\n✅ Silver Job 03 Red Hat Jira completado.")
    print(f"   Output: {output_path}")
    spark.stop()


if __name__ == "__main__":
    main()
