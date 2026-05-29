"""
feature_store.py — ShiftMetrics ML Pipeline
Carga datos desde BigQuery y aplica feature engineering reproducible.
Toda transformación está aquí — ninguna en los scripts de entrenamiento.
"""

import numpy as np
import pandas as pd
from google.cloud import bigquery

from config import (
    GCP_PROJECT, BQ_TABLE, TARGET_COL, FEATURE_COLS,
    TRAIN_END_YEAR, CAL_END_YEAR, VAL_END_YEAR,
)


# 1. Carga desde BigQuery

def load_from_bigquery(table: str = BQ_TABLE, project: str = GCP_PROJECT) -> pd.DataFrame:
    """
    Lee sprint_features desde BigQuery usando Storage API (más rápido que REST).
    Retorna el DataFrame raw sin transformaciones.
    """
    client = bigquery.Client(project=project)
    query = f"""
        SELECT
            sprint_id, project, sprint, sprint_year, sprint_month,
            -- Jira
            num_bugs_sprint, num_stories_sprint, num_tasks_sprint,
            total_issues_sprint, bug_story_ratio, avg_cycle_time_days,
            -- DORA
            deploy_frequency_weekly, change_failure_rate,
            -- CK (1.6% cobertura — solo para ablation study)
            avg_wmc, avg_cbo, avg_rfc, avg_lcom, avg_loc,
            -- Target
            {TARGET_COL}
        FROM `{table}`
        WHERE sprint_year IS NOT NULL
          AND sprint_year BETWEEN 2000 AND 2021
        ORDER BY sprint_year, sprint_month, project
    """
    print(f"Cargando datos desde BigQuery: {table}...")
    df = client.query(query).to_dataframe(
        create_bqstorage_client=True,   # usa BQ Storage API — más rápido
        dtypes={TARGET_COL: "int8"},
    )
    print(f"  {len(df):,} filas, tasa positivos={df[TARGET_COL].mean():.3f}")
    return df


# 2. Feature Engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones al DataFrame raw.
    No modifica el DataFrame original (retorna copia).
    """
    df = df.copy()

    # Indicadores de ausencia (deben ir ANTES de imputacion/transformaciones)
    df["bsr_missing"]   = df["bug_story_ratio"].isna().astype("int8")
    df["cycle_missing"] = df["avg_cycle_time_days"].isna().astype("int8")
    df["dora_missing"]  = df["deploy_frequency_weekly"].isna().astype("int8")

    # Transformaciones log1p: bug_story_ratio (p50=2.5, p99=48),
    # avg_cycle_time_days (p50=90d, p99=2341d, log-normal), total_issues (p50=5, p95=98)
    df["log_avg_cycle_time"] = np.log1p(df["avg_cycle_time_days"].fillna(0))
    df["log_bug_story_ratio"] = np.log1p(df["bug_story_ratio"].fillna(0))
    df["log_total_issues"]    = np.log1p(df["total_issues_sprint"])

    # Encoding ciclico del mes (preserva continuidad enero-diciembre)
    df["sprint_month_sin"] = np.sin(2 * np.pi * df["sprint_month"] / 12)
    df["sprint_month_cos"] = np.cos(2 * np.pi * df["sprint_month"] / 12)

    # Interacciones basadas en dominio: bugs_per_issue normaliza por velocidad del sprint
    df["bugs_per_issue"] = df["num_bugs_sprint"] / df["total_issues_sprint"].clip(lower=1)

    # log_cycle_x_bsr: bug cronico que tarda mucho en resolverse, signal de defecto escapado
    df["log_cycle_x_bsr"] = df["log_avg_cycle_time"] * df["log_bug_story_ratio"]

    return df


# 3. Split temporal

def temporal_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_YEAR,
    cal_end:   int = CAL_END_YEAR,
    val_end:   int = VAL_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide por sprint_year sin leakage temporal — 4 splits.

    Diseño deliberado de splits:
      Train (2000-train_end): HPO training set para Optuna
      Cal   (train_end+1..cal_end): holdout exclusivo para calibración de probabilidades;
            el modelo refit usa train+val (saltando cal), por lo que cal nunca fue
            visto durante entrenamiento — calibración limpia garantizada.
      Val   (cal_end+1..val_end): HPO evaluation + selección de threshold
      Test  (val_end+1..2021): evaluación final intocable

    Con TRAIN_END=2014, CAL_END=2015, VAL_END=2018:
      Train: 2000-2014 -> ~25,400 filas
      Cal:   2015      -> ~1,700  filas
      Val:   2016-2018 -> 12,941  filas
      Test:  2019-2021 ->  6,683  filas
    """
    train = df[df["sprint_year"] <= train_end].copy()
    cal   = df[(df["sprint_year"] > train_end) & (df["sprint_year"] <= cal_end)].copy()
    val   = df[(df["sprint_year"] > cal_end)   & (df["sprint_year"] <= val_end)].copy()
    test  = df[df["sprint_year"] > val_end].copy()

    for name, split in [("Train", train), ("Cal", cal), ("Val", val), ("Test", test)]:
        print(
            f"  {name:<5}: {len(split):>6,} filas "
            f"({len(split)/len(df)*100:.1f}%), "
            f"años {split['sprint_year'].min()}-{split['sprint_year'].max()}, "
            f"pos={split[TARGET_COL].mean():.3f}"
        )
    return train, cal, val, test


# 4. Generador LOPO-CV

def lopo_splits(
    train_df: pd.DataFrame,
    projects: list[str] | None = None,
):
    """
    Leave-One-Project-Out Cross-Validation sobre el train set.
    Genera (X_train, y_train, X_test, y_test, project_name) por fold.

    Proyectos elegidos por balance de clases y tamaño:
      HTTPCLIENT (51.9% pos), IO (61%), MATH (70.2%), MYFACES (78.5%), NET (82.7%)
    """
    from config import LOPO_PROJECTS, FEATURE_COLS

    if projects is None:
        projects = LOPO_PROJECTS

    for project in projects:
        mask_out  = train_df["project"] == project
        mask_in   = ~mask_out

        X_tr = train_df.loc[mask_in,  FEATURE_COLS]
        y_tr = train_df.loc[mask_in,  TARGET_COL]
        X_te = train_df.loc[mask_out, FEATURE_COLS]
        y_te = train_df.loc[mask_out, TARGET_COL]

        if len(X_te) == 0:
            print(f"  LOPO: proyecto {project} no encontrado en train — skip")
            continue

        print(f"  LOPO fold [{project}]: "
              f"train={len(X_tr):,}, test={len(X_te):,} "
              f"(pos={y_te.mean():.2f})")
        yield X_tr, y_tr, X_te, y_te, project


# 5. Pipeline completo

def get_feature_matrix(
    table: str = BQ_TABLE,
    project: str = GCP_PROJECT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Punto de entrada principal:
      1. Carga desde BQ
      2. Aplica feature engineering
      3. Hace el split temporal 4-way
      4. Retorna (train, cal, val, test, df_full)

    Nota sobre el diseño de splits: el modelo refit es entrenado sobre train+val
    (saltando cal), de modo que cal es un holdout limpio para calibración de
    probabilidades — el modelo nunca vio estos datos durante entrenamiento.
    """
    df_raw  = load_from_bigquery(table, project)
    df_feat = engineer_features(df_raw)

    print("\nSplit temporal (4-way):")
    train, cal, val, test = temporal_split(df_feat)

    return train, cal, val, test, df_feat


if __name__ == "__main__":
    train, cal, val, test, df = get_feature_matrix()
    print(f"\nFeature matrix lista: {len(FEATURE_COLS)} features")
    print(f"Columnas: {FEATURE_COLS}")
    print(df[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]])
