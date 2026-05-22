import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_timestamp, split, when, lower, year, month, concat_ws, sum as _sum, count, round

def main():
    spark = SparkSession.builder \
        .appName("ShiftMetrics_Silver_Apache_Jira") \
        .getOrCreate()

    # Rutas de GCS (Bronze a Silver)
    input_path = "gs://shiftmetrics-bronze/apache-jira-parquet/issues/"
    output_path = "gs://shiftmetrics-silver/apache-jira/"

    print(f"Leyendo datos desde: {input_path}")
    
    # 1. Leer las colecciones de issues en formato Parquet
    df = spark.read.parquet(input_path)

    # 2. Filtrar issues sin key o sin created
    df = df.filter(col("key").isNotNull() & col("created").isNotNull())

    # 3. Eliminar columnas que tengan más del 50% de datos vacíos 
    # (según EDA: regression, environment, duedate, etc.)
    cols_to_drop = ["regression", "patchinfo", "environment", "duedate", "timeestimate"]
    existing_cols = df.columns
    cols_to_drop = [c for c in cols_to_drop if c in existing_cols]
    df = df.drop(*cols_to_drop)

    # 4. Parsear fechas y calcular Cycle Time (resolutiondate - created)
    df = df.withColumn("created_ts", to_timestamp("created")) \
           .withColumn("resolution_ts", to_timestamp("resolutiondate"))
    
    df = df.withColumn("cycle_time_days", datediff(col("resolution_ts"), col("created_ts")))

    # 5. Extraer el nombre del proyecto desde la clave (ej: HADOOP-1234 -> HADOOP)
    df = df.withColumn("project", split(col("key"), "-").getItem(0))

    # 6. Clasificar issuetype
    type_col_name = next(
        (c for c in df.columns if c in ("issuetype_name", "issuetype", "issue_type")),
        "issuetype_name"
    )
    
    df = df.withColumn("issue_category", 
                       when(lower(col(type_col_name)).contains("bug"), "bug")
                       .when(lower(col(type_col_name)).contains("story"), "story")
                       .when(lower(col(type_col_name)).contains("task"), "task")
                       .otherwise("other"))

    # 7. Calcular Bug-to-Story Ratio por project + sprint 
    # (Aproximamos 'sprint' por Año-Mes ya que no hay columna nativa de sprint)
    df = df.withColumn("sprint", concat_ws("-", year("created_ts"), month("created_ts")))

    sprint_agg = df.groupBy("project", "sprint").agg(
        _sum(when(col("issue_category") == "bug", 1).otherwise(0)).alias("num_bugs"),
        _sum(when(col("issue_category").isin("story", "task"), 1).otherwise(0)).alias("num_stories_tasks"),
        count("*").alias("total_issues")
    )
    
    sprint_agg = sprint_agg.withColumn(
        "bug_story_ratio", 
        when(col("num_stories_tasks") > 0, round(col("num_bugs") / col("num_stories_tasks"), 3)).otherwise(None)
    )

    # Unimos el ratio de vuelta al DataFrame a nivel de issue
    df_silver = df.join(sprint_agg.select("project", "sprint", "bug_story_ratio"), on=["project", "sprint"], how="left")

    print(f"Escribiendo datos procesados en capa Silver: {output_path}")
    
    # Escribir en formato Parquet particionado por proyecto
    df_silver.write \
        .mode("overwrite") \
        .partitionBy("project") \
        .parquet(output_path)

    print("✅ Job Silver de Apache Jira completado exitosamente.")
    spark.stop()

if __name__ == "__main__":
    main()
