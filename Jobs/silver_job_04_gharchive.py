# =============================================================================
# Silver Job 04 — GHArchive 2022
# ShiftMetrics Analytics · EAFIT 2026
#
# Input:  gs://shiftmetrics-bronze/gharchive/*.json.gz  (24 archivos, 1.69 GiB)
# Output: gs://shiftmetrics-silver/gharchive/  (Parquet particionado por year, month)
#
# Qué hace este job:
#   1. Lee los 24 archivos NDJSON.gz — Spark los descomprime automáticamente
#   2. Filtra SOLO repos de Apache: repo.name.startswith("apache/")
#   3. Aplana los campos anidados (repo.name, actor.login, payload.ref, etc.)
#   4. Extrae dimensiones temporales: year, month
#   5. Extrae apache_project_key para unirse con Jira Silver en Gold
#   6. Calcula Deployment Frequency: PushEvents a main/master por repo/mes
#   7. Calcula Change Failure Rate: PRs sin merge / total PRs closed por repo/mes
#   8. Produce UNA FILA POR (repo_name, year, month) con las métricas DORA
#   9. Escribe Parquet particionado por year, month en Silver
#
# Por qué se agrega aquí y no en Gold:
#   Los eventos crudos son 1.69 GiB. El Gold necesita solo las métricas
#   agregadas (deploy_frequency, change_failure_rate) por repo/mes.
#   Agregar en Silver reduce el volumen ~1000x antes de llegar al JOIN de Gold.
#
# Cómo ejecutar en Dataproc:
#   gcloud dataproc jobs submit pyspark \
#     gs://shiftmetrics-bronze/scripts/silver_job_04_gharchive.py \
#     --cluster=shiftmetrics-cluster \
#     --region=us-central1 \
#     --project=shiftmetrics-analytics
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when,
    to_timestamp, year, month,
    split, upper,
    count, sum as _sum, round as spark_round
)
from pyspark.sql.types import DoubleType

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Ramas que representan un "deployment" en el proxy DORA
# Un push a main o master = una entrega a producción
DEPLOY_BRANCHES = ["refs/heads/main", "refs/heads/master"]

# Semanas promedio por mes (para convertir pushes/mes → pushes/semana)
# 365 días / 12 meses / 7 días = 4.33 semanas/mes
WEEKS_PER_MONTH = 4.33

# Tipos de evento relevantes para métricas DORA
# Filtramos al mínimo necesario para reducir el volumen procesado
RELEVANT_EVENTS = ["PushEvent", "PullRequestEvent"]


def main():
    # -----------------------------------------------------------------------
    # 0. Crear SparkSession
    # -----------------------------------------------------------------------
    spark = SparkSession.builder \
        .appName("ShiftMetrics_Silver_GHArchive") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    input_path  = "gs://shiftmetrics-bronze/gharchive/*.json.gz"
    output_path = "gs://shiftmetrics-silver/gharchive/"

    print("=" * 60)
    print("Silver Job 04 — GHArchive 2022")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Leer los 24 archivos NDJSON comprimidos
    # -----------------------------------------------------------------------
    # GHArchive usa NDJSON: cada línea es un evento JSON completo.
    # spark.read.json() entiende este formato y descomprime .gz automáticamente.
    #
    # Spark infiere el schema leyendo todos los archivos. Los campos anidados
    # (repo, actor, payload) se leen como StructType. Ejemplo:
    #   root
    #    |-- type: string
    #    |-- repo: struct
    #    |    |-- name: string       ← "apache/spark"
    #    |-- actor: struct
    #    |    |-- login: string      ← "username"
    #    |-- payload: struct
    #    |    |-- ref: string        ← "refs/heads/main"
    #    |    |-- action: string     ← "closed"
    #    |    |-- pull_request: struct
    #    |    |    |-- merged: boolean
    #    |-- created_at: string      ← "2022-01-01T00:01:39Z"
    #
    # NOTA: gzip NO es splittable. Cada archivo .gz = 1 tarea Spark.
    # Con 24 archivos tenemos 24 tareas iniciales — aceptable para este volumen.
    df_raw = spark.read.json(input_path)

    print(f"\nSchema inferido por Spark desde GHArchive:")
    df_raw.printSchema()
    print(f"Total eventos crudos: {df_raw.count():,}")

    # -----------------------------------------------------------------------
    # 2. Filtrar SOLO eventos de repositorios Apache
    # -----------------------------------------------------------------------
    # CRÍTICO: sin este filtro calculamos métricas sobre todos los repos de GitHub.
    # EDA_04 demostró que sin filtro el CFR era 77.3% (basura).
    # Con filtro apache/ el CFR es 26.1% (coherente con proyectos open-source).
    #
    # col("repo.name") accede al campo "name" dentro del struct "repo".
    # startswith() filtra strings que comienzan con "apache/".
    # Resultado: "apache/spark" ✅ | "microsoft/vscode" ❌
    df_apache = df_raw.filter(col("repo.name").startswith("apache/"))

    apache_count = df_apache.count()
    print(f"\nEventos de repos Apache: {apache_count:,} "
          f"({apache_count / df_raw.count() * 100:.2f}% del total)")

    # -----------------------------------------------------------------------
    # 3. Filtrar solo eventos relevantes para métricas DORA
    # -----------------------------------------------------------------------
    # De los 13 tipos de evento en Apache repos, solo necesitamos:
    #   PushEvent         → Deployment Frequency
    #   PullRequestEvent  → Change Failure Rate
    # Esto reduce el volumen antes de aplanar los campos.
    df = df_apache.filter(col("type").isin(RELEVANT_EVENTS))

    print(f"Eventos relevantes (Push + PR): {df.count():,}")

    # -----------------------------------------------------------------------
    # 4. Aplanar el schema anidado — extraer solo los campos necesarios
    # -----------------------------------------------------------------------
    # Pasamos de structs anidados a columnas planas.
    # Esto simplifica el código posterior y reduce el tamaño del DataFrame.
    #
    # col("repo.name")                   → "apache/spark"
    # col("actor.login")                 → "username"
    # col("payload.ref")                 → "refs/heads/main" (PushEvent)
    # col("payload.action")              → "closed" (PullRequestEvent)
    # col("payload.pull_request.merged") → true/false (PullRequestEvent)
    #   Si el evento no tiene pull_request (e.g., PushEvent), este campo es null.
    df = df.select(
        col("id").alias("event_id"),
        col("type").alias("event_type"),
        col("repo.name").alias("repo_name"),
        col("actor.login").alias("actor_login"),
        col("payload.ref").alias("payload_ref"),
        col("payload.action").alias("payload_action"),
        col("payload.pull_request.merged").alias("pr_merged"),
        col("created_at")
    )

    # -----------------------------------------------------------------------
    # 5. Parsear created_at y extraer dimensiones temporales
    # -----------------------------------------------------------------------
    # GHArchive usa ISO 8601: "2022-01-01T00:01:39Z"
    # to_timestamp sin formato explícito funciona para este estándar.
    #
    # year() y month() extraen el año y mes para particionar y agregar.
    df = df \
        .withColumn("created_ts", to_timestamp(col("created_at"))) \
        .withColumn("year",  year(col("created_ts"))) \
        .withColumn("month", month(col("created_ts"))) \
        .drop("created_at")    # el timestamp parseado reemplaza al string crudo

    # -----------------------------------------------------------------------
    # 6. Extraer apache_project_key para facilitar el JOIN en Gold
    # -----------------------------------------------------------------------
    # El job Gold necesita unir GHArchive con Apache Jira por proyecto.
    # Apache Jira usa claves como "SPARK", "HADOOP", "AIRFLOW".
    # GHArchive usa nombres como "apache/spark", "apache/hadoop", "apache/airflow".
    #
    # Transformación:
    #   "apache/spark"   → split por "/" → ["apache","spark"] → [1] = "spark" → upper = "SPARK"
    #   "apache/airflow" → "AIRFLOW"
    #   "apache/hbase"   → "HBASE"
    #
    # NOTA: La correspondencia no es 100% perfecta (e.g., "apache/incubator-nuttx"
    # no matchea un proyecto Jira), pero el Gold usará un LEFT JOIN — los no matches
    # simplemente quedarán sin datos DORA, no romperán el pipeline.
    df = df.withColumn(
        "apache_project_key",
        upper(split(col("repo_name"), "/").getItem(1))
    )

    print("\nEjemplo de repo_name → apache_project_key:")
    df.select("repo_name", "apache_project_key") \
      .distinct() \
      .orderBy("repo_name") \
      .show(20, truncate=False)

    # -----------------------------------------------------------------------
    # 7. Calcular Deployment Frequency por repo/mes
    # -----------------------------------------------------------------------
    # Proxy DORA: un push a main o master = un deployment a producción.
    #
    # Filtramos PushEvents donde payload_ref es la rama principal.
    # Agrupamos por (repo_name, year, month) y contamos los pushes.
    # Convertimos a pushes/semana dividiendo por 4.33.
    #
    # EDA_04 encontró: apache/camel=11 pushes, apache/airflow=10 pushes
    # en la muestra de 3 meses. Coherente con proyectos ASF activos.
    deploy_df = df.filter(
        (col("event_type") == "PushEvent") &
        (col("payload_ref").isin(DEPLOY_BRANCHES))
    ).groupBy("repo_name", "apache_project_key", "year", "month").agg(
        count("*").alias("push_count")
    ).withColumn(
        "deploy_frequency_weekly",
        spark_round(
            col("push_count").cast(DoubleType()) / lit(WEEKS_PER_MONTH),
            3
        )
    )

    print(f"\nRepos con Deployment Frequency calculado: {deploy_df.count():,}")
    print("Top 10 por push_count:")
    deploy_df.orderBy(col("push_count").desc()).show(10, truncate=False)

    # -----------------------------------------------------------------------
    # 8. Calcular Change Failure Rate por repo/mes
    # -----------------------------------------------------------------------
    # Proxy DORA: CFR = PRs cerradas SIN merge / total PRs cerradas
    #
    # Una PR cerrada sin merge = cambio que fue rechazado = "failure" del cambio.
    # Una PR cerrada con merge  = cambio exitoso incorporado a la rama principal.
    #
    # Lógica de pr_merged:
    #   true  → PR merged ✅ (cambio exitoso)
    #   false → PR closed without merge ❌ (falla)
    #   null  → evento truncado, tratamos como falla (conservador)
    #
    # EDA_04 encontró CFR=26.1% para repos Apache (era 77.3% sin filtro).
    cfr_df = df.filter(
        (col("event_type") == "PullRequestEvent") &
        (col("payload_action") == "closed")
    ).groupBy("repo_name", "apache_project_key", "year", "month").agg(
        count("*").alias("total_prs_closed"),
        _sum(
            when(
                col("pr_merged").isNull() | (col("pr_merged") == False),
                lit(1)
            ).otherwise(lit(0))
        ).alias("prs_not_merged"),
        _sum(
            when(col("pr_merged") == True, lit(1)).otherwise(lit(0))
        ).alias("prs_merged")
    ).withColumn(
        "change_failure_rate",
        when(
            col("total_prs_closed") > 0,
            spark_round(
                col("prs_not_merged").cast(DoubleType()) / col("total_prs_closed"),
                4
            )
        ).otherwise(lit(None).cast(DoubleType()))
    )

    print(f"\nRepos con CFR calculado: {cfr_df.count():,}")
    print("Muestra de CFR por repo:")
    cfr_df.orderBy("repo_name", "year", "month").show(10, truncate=False)

    # -----------------------------------------------------------------------
    # 9. Unir Deployment Frequency y CFR en una tabla DORA por repo/mes
    # -----------------------------------------------------------------------
    # Usamos OUTER JOIN para incluir repos que solo tienen pushes (sin PRs)
    # o solo tienen PRs (sin pushes a main/master).
    # Esto preserva todos los datos aunque una métrica sea null.
    dora_silver = deploy_df.join(
        cfr_df.select(
            "repo_name", "apache_project_key", "year", "month",
            "total_prs_closed", "prs_merged", "prs_not_merged",
            "change_failure_rate"
        ),
        on=["repo_name", "apache_project_key", "year", "month"],
        how="outer"
    )

    # -----------------------------------------------------------------------
    # 10. Schema final y estadísticas de sanidad
    # -----------------------------------------------------------------------
    print("\nSchema final de Silver GHArchive (una fila por repo/mes):")
    dora_silver.printSchema()

    total_rows = dora_silver.count()
    print(f"\nTotal filas en Silver (combinaciones repo/mes): {total_rows:,}")

    print("\nEstadísticas de deploy_frequency_weekly:")
    dora_silver.select("deploy_frequency_weekly") \
               .filter(col("deploy_frequency_weekly").isNotNull()) \
               .summary("count", "50%", "75%", "90%", "mean", "max") \
               .show()

    print("Estadísticas de change_failure_rate (esperado ~0.261 según EDA_04):")
    dora_silver.select("change_failure_rate") \
               .filter(col("change_failure_rate").isNotNull()) \
               .summary("count", "min", "50%", "mean", "max") \
               .show()

    # -----------------------------------------------------------------------
    # 11. Escribir en Silver particionado por year y month
    # -----------------------------------------------------------------------
    # Particionamos por año y mes porque el job Gold une por ventana temporal.
    # Resultado en GCS:
    #   gs://shiftmetrics-silver/gharchive/year=2022/month=1/part-*.parquet
    #   gs://shiftmetrics-silver/gharchive/year=2022/month=2/part-*.parquet
    #   ...
    #
    # Cuando el Gold necesita datos de marzo 2022, Spark lee solo
    # year=2022/month=3/ — ignora los otros meses (partition pruning).
    dora_silver.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(output_path)

    print(f"\n✅ Silver Job 04 GHArchive completado.")
    print(f"   Output: {output_path}")
    print(f"   Schema producido: repo_name, apache_project_key, year, month,")
    print(f"     push_count, deploy_frequency_weekly,")
    print(f"     total_prs_closed, prs_merged, prs_not_merged, change_failure_rate")
    spark.stop()


if __name__ == "__main__":
    main()
