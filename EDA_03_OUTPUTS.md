# EDA_03 Red Hat Jira — Outputs Validados
**Ejecutado localmente: 2026-05-21**

## Dataset Real
- **Fuente**: `redhat-inputs.zip` (251 proyectos, 251 CSVs)
- **Filas totales**: 505,096 issues
- **RAM en memoria**: ~105 MB
- **Columnas**: `Issue key`, `Issue Type`, `Status`, `Project key`, `Project name`, `Project type`, `Resolution`, `Created`, `Resolved`

## HALLAZGO IMPORTANTE: redhat-outputs.zip
`redhat-outputs.zip` (248 MB) **NO contiene issues Jira**.
Sus CSVs tienen columnas como `['Time', 'beta', 'alpha', 'epsilon', 'gamma', 'm']`
y matrices numéricas de modelo/simulación. **No usar para el pipeline ML.**

## Issue Types
| Tipo | Issues |
|---|---|
| Bug | 223,922 |
| Task | 112,917 |
| Story | 71,406 |
| Feature Request | 33,654 |
| Enhancement | 24,826 |
| Epic | 13,293 |
| Component Upgrade | 9,650 |
| Feature | 6,656 |
| Spike | 3,013 |
| Release | 1,104 |

## Estados
| Estado | Issues |
|---|---|
| Closed | 362,387 |
| Resolved | 61,483 |
| New | 21,621 |
| Open | 13,427 |
| To Do | 8,453 |
| Done | 6,696 |

## Cycle Time (Created → Resolved, dayfirst=True)
- Issues con resolución: **436,475 / 505,096 (86.4%)**
- **p50: 28.0 días**
- **p75: 127.0 días**
- **p90: 400.0 días**
- media: 156.5 días

## Nulos por Columna
| Columna | % Nulos |
|---|---|
| Fix Version/s | 99.96% |
| Assignee | 99.95% |
| Updated | 99.91% |
| Resolved | 13.56% |
| Resolution | 13.39% |
| Created | 0.09% |
| Demás columnas | 0% |

**Columnas a eliminar** (>50% nulos): `Fix Version/s`, `Assignee`, `Updated`

## Parámetros para Dataset Sintético (calibrados)
- `cycle_time_p50`: 28 días
- `cycle_time_p75`: 127 días
- `cycle_time_mean`: 156.5 días → usar `lognormal(mean=log(28), sigma=1.2)` aprox
- `bug_ratio`: Bugs / (Tasks + Stories) = 223922 / (112917 + 71406) = **1.21**
  > Nota: ratio mucho mayor a 1 — Red Hat reporta más bugs que historias (distinto a Apache Jira)
- `issues_resueltos_pct`: 86.4%

## Formato de Fechas Confirmado
- Formato: `DD/MM/YYYY HH:MM` (ej: `22/03/2023 17:04`)
- `dayfirst=True` OBLIGATORIO en todos los `pd.to_datetime()`
