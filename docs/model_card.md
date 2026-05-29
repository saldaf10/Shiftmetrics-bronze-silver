# Model Card — ShiftMetrics Sprint Defect Predictor

**Version**: 2.0.0
**Date**: 2026-Q2
**Authors**: ShiftMetrics Analytics Team — EAFIT MDS&A
**MLflow Experiment**: `shiftmetrics-sprint-defect`
**Model Registry**: `ShiftMetrics-DefectoEscapado` v4 (alias: `champion`, `production`)

---

## 1. Model Details

| Property | Value |
|---|---|
| **Model family** | Gradient Boosted Trees (XGBoost 2.1.3) |
| **Champion selection** | Bayesian HPO (Optuna TPE, 150 trials XGB / 200 trials LGBM); challenger retains title if delta F2 > 0.005 on validation set |
| **Calibration** | Isotonic regression via `CalibratedClassifierCV(cv="prefit")`, fitted on dedicated holdout 2015 (cal set) |
| **Hyperparameter tuning** | Optuna TPE sampler + MedianPruner, persistent SQLite storage (`optuna_shiftmetrics.db`), seed=42 |
| **Primary metric** | F2-score (beta=2): recall weighted 2x precision — aligned with the asymmetric cost of missed defects |
| **Operational threshold** | 0.220 — jointly optimal for F2 and ROI on the validation set |
| **Imbalance handling** | `scale_pos_weight = 12631/30116 = 0.419` (XGBoost internal reweighting) |
| **Temporal split** | 4-way: train (2000-2014) / cal (2015) / val (2016-2018) / test (2019-2021) |

---

## 2. Intended Use

### Primary use case

Predict `defecto_escapado` (binary: a defect escaped to production during or after a sprint) for a given software sprint, evaluated before or at sprint close, to trigger early quality review or release gate escalation.

### Intended users

- Engineering Managers reviewing sprint health dashboards
- QA leads prioritizing pre-release testing effort
- DevOps pipelines integrating automated sprint risk signals into CI/CD gates

### Out-of-scope uses

- Evaluating individual developer quality: the unit of analysis is the sprint, not the person. Predictions must not be repurposed to generate per-developer quality scores.
- Projects with fewer than 3 sprints of history: cold-start conditions produce unreliable predictions. No project-specific embeddings or history features are included.
- Projects outside the Apache JIRA ecosystem: the feature distribution — particularly cycle time and bug story ratio — may differ significantly from the training distribution, requiring recalibration.
- Point-in-time defect cause analysis: this is a predictive model, not a root-cause tool.

---

## 3. Training Data

| Property | Detail |
|---|---|
| **Primary source** | Apache JIRA issues exported to BigQuery (`shiftmetrics_gold.sprint_features`) |
| **Secondary sources** | PROMISE repository (CK metrics, 1.6% coverage); GHArchive DORA signals (9% coverage) |
| **Total records** | 42,747 sprint-records across 42 Apache projects |
| **Temporal scope** | Sprint years 2000-2021 |
| **Train split** | 2000-2014 — approximately 25,400 rows; used exclusively for Optuna HPO |
| **Calibration split** | 2015 — approximately 1,700 rows; held out from all model training; used only to fit the probability calibrator |
| **Validation split** | 2016-2018 — 12,941 rows; HPO evaluation target and threshold selection |
| **Test split** | 2019-2021 — 6,683 rows; final evaluation, never touched during any training or selection step |
| **Split method** | Strict temporal split by `sprint_year`; no random shuffling; no data from future years ever visible during training |
| **Positive rate** | 70.45% (30,116 positive / 12,631 negative) — moderate class imbalance toward the positive class |
| **Imbalance strategy** | XGBoost `scale_pos_weight=0.419`; Logistic Regression `class_weight="balanced"` + SMOTE evaluated in grid search |

---

## 4. Features

| Feature | Group | Engineering | Coverage | Justification |
|---|---|---|---|---|
| `num_bugs_sprint` | Jira counts | Raw count | 100% | Primary SHAP driver: direct count of bugs in the sprint |
| `num_stories_sprint` | Jira counts | Raw count | 100% | Sprint velocity proxy |
| `num_tasks_sprint` | Jira counts | Raw count | 100% | Sprint workload composition |
| `total_issues_sprint` | Jira counts | Raw count | 100% | Sprint size normalizer |
| `log_avg_cycle_time` | Process | log1p(avg_cycle_time_days) | 91.8% | p50=90d, p99=2341d — log-normal; second-ranked SHAP feature |
| `log_bug_story_ratio` | Process | log1p(bug_story_ratio) | 37.5% | p50=2.5, p99=48 — heavy right skew; basis for BSR baseline |
| `log_total_issues` | Process | log1p(total_issues_sprint) | 100% | Variance stabilization for sprint size |
| `bugs_per_issue` | Interaction | num_bugs / total_issues (clipped at 1) | 100% | Bug density normalized by sprint velocity; third SHAP rank |
| `log_cycle_x_bsr` | Interaction | log_cycle_time x log_bsr | 100% | Chronic bug backlog signal: long-running bugs with high BSR |
| `sprint_year` | Temporal | Raw integer | 100% | Captures secular drift: cycle_time fell ~96% from 2008 to 2021 |
| `sprint_month_sin` | Temporal | sin(2pi * month / 12) | 100% | Cyclical month encoding preserving January-December continuity |
| `sprint_month_cos` | Temporal | cos(2pi * month / 12) | 100% | Second component of cyclical encoding |
| `deploy_frequency_weekly` | DORA | Raw | ~9% | GHArchive DORA proxy; sparse but informative when present |
| `change_failure_rate` | DORA | Raw | ~9% | GHArchive DORA proxy |
| `bsr_missing` | Missing indicator | 1 if bug_story_ratio is NaN | — | 62.5% of sprints lack BSR; absence encodes sprint type (no bugs or no stories) |
| `cycle_missing` | Missing indicator | 1 if avg_cycle_time is NaN | — | 8.2% missing; fourth SHAP rank — process immaturity signal |
| `dora_missing` | Missing indicator | 1 if deploy_freq is NaN | — | 91% missing; GHArchive coverage limited to ~20 projects |

**Note on CK metrics**: `avg_wmc`, `avg_cbo`, `avg_rfc`, `avg_lcom`, `avg_loc` are available for only 1.6% of sprints (PROMISE dataset has no overlap with Apache JIRA). They are excluded from the main feature set but retained in BigQuery for future ablation study. Their SHAP contribution is negligible in the current dataset.

---

## 5. Evaluation Results

### 5.1 Baseline comparison (validation set)

All metrics reported at optimal decision threshold per model. For baselines, the implicit threshold is predict-all-positive (MajorityClass) or BSR > train-optimal cutoff (BSR). Champion metrics reported at t=0.220.

| Model | Val F2 | Val PR-AUC | Val Recall | Val Precision | Val Brier |
|---|---|---|---|---|---|
| Baseline: Majority Class (predict-all-1) | 0.916 | 0.686 | 1.000 | 0.686 | 0.217 |
| Baseline: BSR threshold (auto, train-optimal) | 0.916 | 0.828 | 1.000 | 0.686 | 0.191 |
| Best Logistic Regression (C=10, L2, SMOTE) | 0.861 | 0.977 | 0.839 | 0.959 | 0.100 |
| XGBoost, Optuna 150 trials (uncalibrated, t=0.5) | 0.969 | 0.984 | 0.998 | 0.875 | 0.106 |
| **Champion: XGBoost + isotonic (calibrated, t=0.220)** | **0.970** | **0.984** | **0.997** | **0.873** | **0.061** |

**Interpretation**: The majority class baseline achieves F2=0.916 by exploiting the 70.45% positive rate with a trivial predict-all-1 strategy. It does so at the cost of zero precision — every sprint is flagged. The champion model surpasses this baseline on F2 while maintaining precision of 87.3%, meaning only 12.7% of flagged sprints are false positives. The logistic regression challenger does not close the gap on F2 (0.861 vs 0.970 = delta of 10.9pp), confirming the non-linear interactions captured by gradient boosting are essential for this domain.

### 5.2 Final evaluation (test set, 2019-2021)

| Metric | Value | 95% Bootstrap CI (n=1000) |
|---|---|---|
| F2-score | 0.9655 | [0.9627, 0.9684] |
| Recall | 0.9953 | [0.9932, 0.9972] |
| Precision | 0.8623 | — |
| PR-AUC | 0.9791 | — |
| Brier score | 0.0666 | — |
| Flagging rate | 74.2% | — |

**Validation-to-test generalization**: Val F2=0.9698 to test F2=0.9655 represents a degradation of 0.43 percentage points — well within the bootstrap CI width of 0.57pp. Val Brier=0.0605 to test Brier=0.0666 shows a calibration cost of 0.006, consistent with mild distribution shift in 2019-2021 relative to 2016-2018 (confirmed by temporal drift analysis). Both gaps are within acceptable bounds for temporal generalization.

### 5.3 LOPO-CV stability (5 projects)

Two evaluation modes: base model at t=0.5 (measures generalizability of the raw model), and calibrated model at t=0.220 (deployment operating point).

| Project | Pos. rate | F2 (base, t=0.5) | F2 (calibrated, t=0.220) |
|---|---|---|---|
| HTTPCLIENT | 61.0% | 0.8879 | 0.8925 |
| IO | 62.0% | 0.9524 | 0.9545 |
| MATH | 72.0% | 0.9586 | 0.9605 |
| MYFACES | 93.0% | 0.9858 | 0.9858 |
| NET | 84.0% | 0.9781 | 0.9796 |
| **Mean** | | **0.9526 +/- 0.039** | **0.9546 +/- 0.037** |

HTTPCLIENT shows the lowest cross-project F2 (0.8879-0.8925), attributable to its lower positive rate (61%) relative to the training distribution mean (70.45%). Models trained on predominantly positive data tend to be overconfident in the positive direction; at lower base rates, precision suffers disproportionately. This is an acknowledged limitation for deployment in projects with sparse defect history.

### 5.4 Model comparison significance

McNemar test (XGBoost vs LightGBM, validation set, t=0.5): chi2=645.59, p < 0.001. The difference between models is statistically significant. Although LightGBM corrects 976 errors exclusive to XGBoost while XGBoost corrects only 130 exclusive to LightGBM at t=0.5, this is a threshold-specific result. Optuna optimizes each trial at its own threshold; XGBoost's best trial achieves F2=0.9697 vs LightGBM's 0.8975 — a gap of 7.22pp — by maximizing recall more aggressively, which is the correct behavior under a F2 objective. The McNemar result and the champion selection are not contradictory: they operate at different operating points.

---

## 6. Known Limitations

1. **Majority class dominance at F2**: The dataset's 70.45% positive rate means predict-all-1 achieves F2=0.916. The champion's advantage is operationally meaningful (87% precision vs 69%) but the F2 gap is moderate. In deployments where False Positives carry significant operational cost, a higher threshold may be necessary.

2. **HTTPCLIENT cross-project gap**: LOPO F2=0.887 for HTTPCLIENT (61% positive rate) vs 0.986 for MYFACES (93%). Deployments in projects with positive rates substantially below 70% may see lower precision.

3. **Temporal concept drift confirmed**: KS + PSI test flagged 4 of 17 features as drifted between train (2000-2014) and test (2019-2021). Most notable: `log_avg_cycle_time` (cycle time fell from ~398 days in 2008 to ~15 days in 2021 — a 96% drop). The feature `sprint_year` partially compensates by capturing the secular trend, but the model should be retrained annually.

4. **DORA and CK coverage**: `deploy_frequency_weekly`, `change_failure_rate` (DORA) are null for ~91% of records. CK static metrics are null for 98.4% of records. Their practical contribution is captured through the `dora_missing` indicator but the actual DORA values add limited signal in the current dataset.

5. **Cold-start**: Projects with fewer than 5 sprints produce high-variance predictions. No project-specific prior or embedding is included. Predictions for new projects should be treated with explicit uncertainty flagging.

6. **Flagging rate**: At t=0.220, 74.2% of sprints are flagged as high-risk. In operational contexts where review capacity is limited, this rate may require threshold recalibration or a tiered risk system (high/medium/low) rather than a binary flag.

---

## 7. Ethical Considerations

- **Unit of analysis is the sprint, not the person**: Model predictions reflect sprint-level process and issue patterns. They must not be used to infer individual developer quality, productivity, or reliability.
- **Asymmetric cost design**: The F2 objective (beta=2) and the ROI model (FN costs 3x FP) are calibrated to accept more false positives in exchange for fewer missed defects. This is appropriate for quality assurance contexts where under-reporting is more costly than over-reporting.
- **Fairness by project type**: LOPO-CV across 5 projects spanning 61%-93% positive rates confirms no systematic breakdown by project type within the Apache ecosystem. No claims are made for projects outside this distribution.
- **Transparency**: All experiment runs, artifact logs, model signatures, and calibration curves are available in the MLflow tracking server. The model registry maintains full version history.

---

## 8. Operational Thresholds

Both F2-optimal and ROI-optimal thresholds converge at the same value, providing a coherent operating point.

| Threshold type | Value | Val F2 | Val Recall | Val Precision | Rationale |
|---|---|---|---|---|---|
| **F2-optimal** | 0.220 | 0.9698 | 0.9974 | 0.8732 | Maximizes F2 on validation set; searched over [0.05, 0.95] in steps of 0.01 |
| **ROI-optimal** | 0.220 | 0.9698 | 0.9974 | 0.8732 | Maximizes net value = TP - 1xFP - 3xFN on validation set; cost_FN=3x cost_FP |

Both thresholds were selected on the validation set exclusively and validated on the test set without re-selection. The convergence of F2 and ROI optima at the same threshold (0.220) confirms that the economic model and the statistical objective are aligned: catching escaped defects at the cost of some false positives is the dominant strategy under both criteria.

**Monitoring guards**:
- Flagging rate > 85%: recheck calibration or raise threshold
- Test Brier > 0.10: trigger recalibration cycle
- LOPO-CV F2 std > 0.05 on new project: flag for project-specific threshold adjustment

---

## 9. Infrastructure

| Component | Detail |
|---|---|
| **MLflow tracking** | Cloud Run (us-central1), `https://mlflow-server-919593201130.us-central1.run.app` |
| **Artifact store** | `gs://shiftmetrics-bronze/mlruns/` |
| **Training environment** | Vertex AI Workbench `shiftmetrics-ml` (n1-standard-8, us-central1-a) |
| **Feature store** | BigQuery `shiftmetrics-analytics.shiftmetrics_gold.sprint_features` |
| **Model serialization** | XGBoost native + scikit-learn wrapper (`CalibratedClassifierCV`) via joblib/mlflow.sklearn |
| **HPO storage** | SQLite persistent (`optuna_shiftmetrics.db`) — studies are resumable across sessions |

---

## 10. Versioning and Reproducibility

- All random seeds: `RANDOM_SEED=42` (Python, NumPy, XGBoost, LightGBM, Optuna TPE sampler)
- Temporal split boundaries hard-coded in `config.py`: `TRAIN_END_YEAR=2014`, `CAL_END_YEAR=2015`, `VAL_END_YEAR=2018`
- Optuna studies are deterministic given the same SQLite history; results are reproducible if the same study file is present
- Full pipeline reproducible via `python run_pipeline.py --n-trials 50`
- MLflow run IDs for the canonical run (v4):
  - Champion selection: `b81e450774924dfc94cb82bd173cc2ef`
  - Calibration: `355c43bed63b4c7e96d9afd37eacbd4b`
  - LOPO-CV: `cc338b317d73472c96cfaa1b9f6d0a86`
  - SHAP: `0e6c9b9597cf4023a0b9a1976fd6c05f`
  - Drift: `d536b2fdbac74a798ad0a2e8b5081b4e`
  - Threshold: `4f398f29acdb4d92b5710aef391b08ed`
  - Final evaluation: `90d0407fefeb449a8b99ea3c6dc3f8dc`

---

*This model card follows the structure proposed by Mitchell et al. (2019) "Model Cards for Model Reporting." All quantitative claims are derived from logged metrics in the MLflow tracking server and are reproducible from the canonical pipeline run.*
