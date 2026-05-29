"""
drift.py — ShiftMetrics ML Pipeline
Análisis de concept drift: temporal real (year-by-year) + simulado (+20%/+50%).
KS tests, PSI, y monitoreo de performance por cohorte temporal.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import mlflow

from config import (
    FEATURE_COLS, TARGET_COL,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
)
from metrics import compute_metrics


# Population Stability Index
def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI (Population Stability Index).
    PSI < 0.1: estable, 0.1-0.2: alerta, > 0.2: drift significativo.
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = (exp_counts / len(expected)).clip(1e-6)
    act_pct = (act_counts / len(actual)).clip(1e-6)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# KS test por feature
def ks_drift_report(
    train: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    KS test de cada feature entre train y target (val/test o cohorte temporal).
    Retorna DataFrame con ks_stat, p_value, psi, drift_flag.
    """
    rows = []
    for feat in features:
        tr = train[feat].dropna().values
        tg = target[feat].dropna().values
        if len(tr) < 10 or len(tg) < 10:
            continue
        ks_stat, p_val = stats.ks_2samp(tr, tg)
        psi_val = psi(tr, tg)
        rows.append({
            "feature":    feat,
            "ks_stat":    round(ks_stat, 4),
            "p_value":    round(p_val, 4),
            "psi":        round(psi_val, 4),
            "drift_flag": int(p_val < 0.05 and psi_val > 0.1),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# Drift temporal real (year-by-year)
def temporal_drift_analysis(
    df_full: pd.DataFrame,
    model,
    features: list[str] = FEATURE_COLS,
    save_dir: str = "/tmp",
) -> pd.DataFrame:
    """
    Para cada año en df_full, evalúa el modelo y registra performance + drift.
    Detecta el año en que el model degradó más de 0.05 en F2 respecto al train.
    """
    df_full = df_full.copy()
    X_all = df_full[features].values
    y_all = df_full[TARGET_COL].values

    # Referencia: tomar train set (años <= 2015)
    train_mask = df_full["sprint_year"] <= 2015
    train_ref  = df_full[train_mask]

    rows = []
    for year in sorted(df_full["sprint_year"].unique()):
        mask   = df_full["sprint_year"] == year
        subset = df_full[mask]
        if len(subset) < 30:
            continue

        X_s = subset[features].values
        y_s = subset[TARGET_COL].values
        prob = model.predict_proba(X_s)[:, 1]
        pred = model.predict(X_s)
        m    = compute_metrics(y_s, pred, prob)

        ks_df   = ks_drift_report(train_ref, subset, features[:5])  # top-5 features
        psi_bsr = psi(
            train_ref["log_bug_story_ratio"].dropna().values,
            subset["log_bug_story_ratio"].dropna().values,
        ) if "log_bug_story_ratio" in subset.columns else 0.0

        rows.append({
            "year":        int(year),
            "n_sprints":   len(subset),
            "pos_rate":    round(float(y_s.mean()), 4),
            "f2":          m["f2"],
            "pr_auc":      m["pr_auc"],
            "brier":       m["brier"],
            "psi_bsr":     round(psi_bsr, 4),
            "n_drift_feat": int(ks_df["drift_flag"].sum()),
        })

    drift_df = pd.DataFrame(rows)

    # Plot: F2 y PSI_bsr por año
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(drift_df["year"], drift_df["f2"], "b-o", label="F2")
    ax1.axhline(0.923, color="gray", linestyle="--", alpha=0.5, label="Baseline F2=0.923")
    ax1.set_ylabel("F2 Score")
    ax1.set_title("Temporal Performance Drift — ShiftMetrics Sprint Defect")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(drift_df["year"], drift_df["psi_bsr"], alpha=0.7, color="orange", label="PSI log_bsr")
    ax2.axhline(0.1, color="red",    linestyle="--", alpha=0.7, label="PSI alert (0.1)")
    ax2.axhline(0.2, color="darkred",linestyle="--", alpha=0.7, label="PSI drift (0.2)")
    ax2.set_xlabel("Sprint Year")
    ax2.set_ylabel("PSI log_bug_story_ratio")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    drift_plot_path = f"{save_dir}/temporal_drift.png"
    plt.savefig(drift_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    drift_df["plot_path"] = drift_plot_path
    return drift_df, drift_plot_path


# Drift simulado (+20% / +50% cycle_time)
def simulated_drift_analysis(
    model,
    test: pd.DataFrame,
    feature_to_shift: str = "log_avg_cycle_time",
    shifts: list[float] = [0.0, 0.20, 0.50],
    save_dir: str = "/tmp",
) -> pd.DataFrame:
    """
    Simula drift en `feature_to_shift` aplicando shifts multiplicativos.
    Compara F2 y Brier con el test set original.
    Justificación: cycle_time real bajó 398->15 días (2008->2021); simular rebote.
    """
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].values

    rows = []
    for shift in shifts:
        X_shifted = X_test.copy()
        X_shifted[feature_to_shift] = X_test[feature_to_shift] * (1 + shift)

        prob = model.predict_proba(X_shifted.values)[:, 1]
        pred = model.predict(X_shifted.values)
        m    = compute_metrics(y_test, pred, prob)

        rows.append({
            "shift_pct":    int(shift * 100),
            "feature":      feature_to_shift,
            "f2":           m["f2"],
            "pr_auc":       m["pr_auc"],
            "brier":        m["brier"],
            "recall":       m["recall"],
            "precision":    m["precision"],
        })
        print(f"  Shift +{int(shift*100)}% {feature_to_shift}: "
              f"F2={m['f2']:.4f}, Brier={m['brier']:.4f}")

    sim_df = pd.DataFrame(rows)

    # Bar chart degradación
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(shifts))
    ax.bar(x - 0.2, sim_df["f2"],    0.35, label="F2",    color="steelblue")
    ax.bar(x + 0.2, sim_df["brier"], 0.35, label="Brier", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{int(s*100)}%" for s in shifts])
    ax.set_xlabel(f"Shift in {feature_to_shift}")
    ax.set_title("Simulated Concept Drift — Performance Degradation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    sim_plot_path = f"{save_dir}/simulated_drift.png"
    plt.savefig(sim_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return sim_df, sim_plot_path


# Main entry point
def run_drift_analysis(
    model,
    champion_name: str,
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    df_full: pd.DataFrame,
) -> dict:
    """
    Análisis completo de drift: temporal real + simulado.
    Loguea resultados a MLflow y retorna dict de resultados.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print(f"\n[Drift] Analizando drift para {champion_name}...")

    with mlflow.start_run(run_name=f"drift_{champion_name}"):
        mlflow.set_tag("model_role", "drift_analysis")
        mlflow.set_tag("champion_name", champion_name)

        # KS report: train vs test
        ks_df = ks_drift_report(train, test, FEATURE_COLS)
        ks_csv = "/tmp/ks_drift_report.csv"
        ks_df.to_csv(ks_csv, index=False)
        mlflow.log_artifact(ks_csv, artifact_path="drift")
        mlflow.log_metric("n_drifted_features", int(ks_df["drift_flag"].sum()))
        print(f"  KS drift: {ks_df['drift_flag'].sum()} / {len(ks_df)} features drifted")

        # Temporal drift
        drift_df, drift_plot = temporal_drift_analysis(df_full, model)
        mlflow.log_artifact(drift_plot, artifact_path="drift")
        temporal_csv = "/tmp/temporal_drift.csv"
        drift_df.to_csv(temporal_csv, index=False)
        mlflow.log_artifact(temporal_csv, artifact_path="drift")

        # Año de mayor caída en F2
        if len(drift_df) > 1:
            drift_df_no_plot = drift_df.drop(columns="plot_path", errors="ignore")
            worst_year = drift_df_no_plot.loc[drift_df_no_plot["f2"].idxmin(), "year"]
            mlflow.log_metric("worst_year_f2", float(drift_df_no_plot["f2"].min()))
            mlflow.log_param("worst_drift_year", int(worst_year))

        # Simulado
        sim_df, sim_plot = simulated_drift_analysis(model, test)
        mlflow.log_artifact(sim_plot, artifact_path="drift")
        sim_csv = "/tmp/simulated_drift.csv"
        sim_df.to_csv(sim_csv, index=False)
        mlflow.log_artifact(sim_csv, artifact_path="drift")

        f2_base = sim_df.loc[sim_df["shift_pct"] == 0, "f2"].values[0]
        f2_50   = sim_df.loc[sim_df["shift_pct"] == 50, "f2"].values[0]
        mlflow.log_metric("f2_shift0",  float(f2_base))
        mlflow.log_metric("f2_shift50", float(f2_50))
        mlflow.log_metric("f2_drop_50pct", float(f2_base - f2_50))

    return {
        "ks_report":    ks_df,
        "temporal_df":  drift_df,
        "simulated_df": sim_df,
        "drift_plots":  [drift_plot, sim_plot],
    }


if __name__ == "__main__":
    from feature_store import get_feature_matrix
    from train_gbm import tune_xgboost

    train, val, test, df_full = get_feature_matrix()
    model, _ = tune_xgboost(train, val, n_trials=3)

    results = run_drift_analysis(model, "XGBoost", train, val, test, df_full)
    print("\nTop features con drift (KS + PSI):")
    print(results["ks_report"].head(5).to_string(index=False))
