# =============================================================================
# Silver Job 01 — PROMISE (Métricas CK)
# ShiftMetrics Analytics · EAFIT 2026
#
# Input:  gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*.csv
# Output: gs://shiftmetrics-silver/promise/  (Parquet particionado por project)
#
# Qué hace este job:
#   1. Lee todos los CSVs de métricas CK del bucket Bronze
#   2. Extrae project y version del nombre del archivo
#   3. Estandariza nombres de columnas a lowercase
#   4. Castea las 18 métricas CK a DoubleType
#   5. Filtra filas sin columna target (bug)
#   6. Binariza bug: (bug > 0) → defect_flag = 1
#   7. Calcula defect_density por proyecto
#   8. Escribe Parquet particionado por project en Silver
#
# Cómo ejecutar en Dataproc:
#   gcloud dataproc jobs submit pyspark \
#     gs://shiftmetrics-bronze/scripts/silver_job_01_promise.py \
#     --cluster=shiftmetrics-cluster \
#     --region=us-central1 \
#     --project=shiftmetrics-analytics
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit,
    input_file_name, regexp_extract,
    count, sum as _sum, round as spark_round
)
from pyspark.sql.types import DoubleType, IntegerType

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Las 18 métricas CK (Chidamber & Kemerer) presentes en bug-data/
# Confirmadas en EDA_01: todos los CSVs tienen estas columnas en común
CK_METRICS = [
    'wmc',   # Weighted Methods per Class — complejidad total de la clase
    'dit',   # Depth of Inheritance Tree — profundidad de herencia
    'noc',   # Number of Children — subclases directas
    'cbo',   # Coupling Between Objects — acoplamiento entre clases
    'rfc',   # Response For a Class — métodos invocables desde la clase
    'lcom',  # Lack of Cohesion in Methods — falta de cohesión
    'ca',    # Afferent Coupling — cuántas clases dependen de ésta
    'ce',    # Efferent Coupling — cuántas clases usa ésta
    'npm',   # Number of Public Methods
    'lcom3', # Variante alternativa de LCOM
    'loc',   # Lines of Code
    'dam',   # Data Access Metric
    'moa',   # Measure of Aggregation
    'mfa',   # Measure of Functional Abstraction
    'cam',   # Cohesion Among Methods
    'ic',    # Inheritance Coupling
    'cbm',   # Coupling Between Methods
    'amc',   # Average Method Complexity
]

# Nombres posibles de la columna target en distintos CSVs del dataset PROMISE
# (el EDA mostró que se llama 'bug' en bug-data/, pero lo hacemos robusto)
TARGET_CANDIDATES = ['bug', 'bugs', 'defects', 'defect', 'class', 'fault']


def main():
    # -----------------------------------------------------------------------
    # 0. Crear SparkSession
    # -----------------------------------------------------------------------
    # getOrCreate() reutiliza una sesión existente si Dataproc ya tiene una,
    # o crea una nueva. El appName aparece en la UI de YARN/Dataproc.
    spark = SparkSession.builder \
        .appName("ShiftMetrics_Silver_PROMISE") \
        .getOrCreate()

    # Silenciar logs INFO de Spark para que el output sea más legible
    spark.sparkContext.setLogLevel("WARN")

    input_path  = "gs://shiftmetrics-bronze/promise/PROMISE-backup/bug-data/*/*.csv"
    output_path = "gs://shiftmetrics-silver/promise/"

    print("=" * 60)
    print("Silver Job 01 — PROMISE")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Leer todos los CSVs del bucket Bronze
    # -----------------------------------------------------------------------
    # - header=true:      la primera fila de cada CSV contiene los nombres de columna
    # - inferSchema=true: Spark deduce el tipo de cada columna leyendo una muestra
    # - mergeSchema=true: si distintos CSVs tienen columnas distintas, las une
    #                     añadiendo null donde la columna no existe en ese archivo
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("mergeSchema", "true") \
        .csv(input_path)

    print(f"\nSchema leído desde Bronze:")
    df.printSchema()

    # -----------------------------------------------------------------------
    # 2. Agregar el path del archivo de origen como columna auxiliar
    # -----------------------------------------------------------------------
    # input_file_name() devuelve la ruta completa del archivo GCS desde el que
    # viene cada fila. Ejemplo:
    #   gs://shiftmetrics-bronze/promise/.../bug-data/ant-1.7.csv
    # La usamos para extraer project y version, y luego la eliminamos.
    df = df.withColumn("_file_path", input_file_name())

    # -----------------------------------------------------------------------
    # 3. Extraer nombre base del archivo
    # -----------------------------------------------------------------------
    # regexp_extract(columna, regex, grupo) devuelve el grupo de captura.
    # Regex:  ([^/]+)\.csv$
    #   [^/]+   → uno o más caracteres que no sean "/" (el nombre del archivo)
    #   \.csv$  → seguido de ".csv" al final del string
    # Resultado: "ant-1.7.csv" → "ant-1.7"
    df = df.withColumn(
        "_basename",
        regexp_extract(col("_file_path"), r"([^/]+)\.csv$", 1)
    )

    # -----------------------------------------------------------------------
    # 4. Extraer el nombre del proyecto desde el basename
    # -----------------------------------------------------------------------
    # Los archivos PROMISE siguen el patrón: {project}-{version}.csv
    # Ejemplos: ant-1.7 | log4j-1.2 | xalan-2.7 | prop-6
    #
    # Regex:  ^(.+?)-\d
    #   ^       → inicio del string
    #   (.+?)   → captura mínima (non-greedy) — se detiene lo antes posible
    #   -\d     → hasta encontrar un guión seguido de un dígito (inicio de versión)
    #
    # Por qué non-greedy (.+?) y no greedy (.+):
    #   Con greedy, "log4j-1.2" → grupo 1 = "log4j-1" (toma de más)
    #   Con non-greedy,          → grupo 1 = "log4j"   (correcto)
    df = df.withColumn(
        "project",
        regexp_extract(col("_basename"), r"^(.+?)-\d", 1)
    )

    # -----------------------------------------------------------------------
    # 5. Extraer la versión del archivo
    # -----------------------------------------------------------------------
    # Regex:  -(\d[\d\.]*)$
    #   -       → guión separador
    #   (       → inicio de grupo de captura
    #   \d      → primer dígito obligatorio
    #   [\d\.]* → seguido de más dígitos o puntos (e.g. "1.7", "2", "10.3")
    #   )       → fin de grupo
    #   $       → al final del string
    # Ejemplos: "ant-1.7" → "1.7" | "prop-6" → "6" | "log4j-1.2" → "1.2"
    df = df.withColumn(
        "version",
        regexp_extract(col("_basename"), r"-(\d[\d\.]*)$", 1)
    )

    # -----------------------------------------------------------------------
    # 6. Estandarizar nombres de columnas a lowercase
    # -----------------------------------------------------------------------
    # Algunos CSVs pueden tener "WMC" en mayúsculas o "Bug" con B mayúscula.
    # Normalizamos todo para que las búsquedas posteriores sean consistentes.
    cols_normalized = {c: c.lower().strip() for c in df.columns}
    for original, normalized in cols_normalized.items():
        if original != normalized:
            df = df.withColumnRenamed(original, normalized)

    # -----------------------------------------------------------------------
    # 7. Castear las métricas CK a DoubleType
    # -----------------------------------------------------------------------
    # inferSchema las leyó como Int o String en algunos casos.
    # Forzamos Double para que el job Gold pueda calcular promedios sin problemas.
    cols_in_df = [c.lower() for c in df.columns]
    for metric in CK_METRICS:
        if metric in cols_in_df:
            df = df.withColumn(metric, col(metric).cast(DoubleType()))

    # -----------------------------------------------------------------------
    # 8. Detectar la columna target dinámicamente
    # -----------------------------------------------------------------------
    # El EDA_01 confirmó que en bug-data/ la columna se llama 'bug'.
    # Pero lo hacemos robusto por si algún CSV la llama distinto.
    target_col = None
    for candidate in TARGET_CANDIDATES:
        if candidate in [c.lower() for c in df.columns]:
            target_col = candidate
            break

    if target_col is None:
        raise ValueError(
            f"No se encontró columna target. "
            f"Candidatos buscados: {TARGET_CANDIDATES}. "
            f"Columnas disponibles: {df.columns}"
        )

    print(f"\nColumna target detectada: '{target_col}'")

    # -----------------------------------------------------------------------
    # 9. Filtrar filas donde la columna target sea null
    # -----------------------------------------------------------------------
    # EDA_01 no encontró nulos en bug-data/, pero lo aplicamos por robustez.
    # Un módulo sin etiqueta no aporta información al modelo.
    before = df.count()
    df = df.filter(col(target_col).isNotNull())
    after  = df.count()
    print(f"Filas eliminadas por target null: {before - after:,} "
          f"(de {before:,} → quedan {after:,})")

    # -----------------------------------------------------------------------
    # 10. Binarizar la columna target → defect_flag
    # -----------------------------------------------------------------------
    # En PROMISE, 'bug' puede ser:
    #   - Un entero: 0 = limpio, 1+ = defectuoso (número de bugs encontrados)
    #   - A veces: "TRUE"/"FALSE" (por eso casteamos a Double primero)
    #
    # Regla: si bug > 0 → defect_flag = 1 (módulo defectuoso)
    #         bug == 0 → defect_flag = 0 (módulo limpio)
    #
    # Guardamos también el valor original en 'bug_count' para referencia.
    df = df.withColumn("bug_count", col(target_col).cast(DoubleType()))
    df = df.withColumn(
        "defect_flag",
        when(col("bug_count") > 0, lit(1))
        .otherwise(lit(0))
        .cast(IntegerType())
    )

    # -----------------------------------------------------------------------
    # 11. Calcular Defect Density por proyecto
    # -----------------------------------------------------------------------
    # Defect Density = módulos defectuosos / total módulos del proyecto
    # Esta métrica calibra el Dataset Sintético (Fase 4 del proyecto).
    #
    # Calculamos en un DataFrame de agregación y hacemos join al principal.
    # No calculamos por sprint porque PROMISE no tiene columna temporal —
    # cada CSV es una versión completa del proyecto.
    density_agg = df.groupBy("project", "version").agg(
        count("*").alias("total_modules"),
        _sum("defect_flag").alias("defective_modules")
    ).withColumn(
        "defect_density",
        spark_round(
            col("defective_modules").cast(DoubleType()) / col("total_modules"),
            4
        )
    )

    print("\nDefect density por proyecto/versión:")
    density_agg.orderBy("project", "version").show(50, truncate=False)

    # Join al DataFrame principal para que cada fila tenga su defect_density
    df = df.join(
        density_agg.select("project", "version", "defect_density", "total_modules"),
        on=["project", "version"],
        how="left"
    )

    # -----------------------------------------------------------------------
    # 12. Eliminar columnas auxiliares de extracción de metadatos
    # -----------------------------------------------------------------------
    df = df.drop("_file_path", "_basename")

    # -----------------------------------------------------------------------
    # 13. Verificar schema final antes de escribir
    # -----------------------------------------------------------------------
    print("\nSchema final a escribir en Silver:")
    df.printSchema()

    # -----------------------------------------------------------------------
    # 14. Escribir en Silver como Parquet particionado por 'project'
    # -----------------------------------------------------------------------
    # - mode("overwrite"): si ya existe data en Silver la reemplaza completamente.
    #   Útil para re-runs. Si quisiéramos append usar mode("append").
    # - partitionBy("project"): crea subcarpetas por proyecto en GCS.
    #   Resultado: gs://shiftmetrics-silver/promise/project=ant/part-*.parquet
    #                                                project=camel/...
    #   Esto permite al job Gold leer solo el proyecto que necesita (partition pruning).
    df.write \
        .mode("overwrite") \
        .partitionBy("project") \
        .parquet(output_path)

    print(f"\n✅ Silver Job 01 PROMISE completado.")
    print(f"   Output: {output_path}")
    spark.stop()


if __name__ == "__main__":
    main()
