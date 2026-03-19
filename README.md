# Evaluation of statistical and machine learning models — An Empirical Approach

> Bias–Variance Tradeoff via Monte Carlo Simulations & Empirical Validation on the Diabetes Dataset (SAS)
> 
> **Authors:** Aya Wifak · Alicia Dahmani · Aya El Farfar — M1 Économétrie & Statistiques, Paris 1 Panthéon-Sorbonne
> **Supervisor:** Philippe De Peretti — Academic Year 2025–2026

---

## Project Overview

This project benchmarks **variable selection methods** across two complementary approaches:
```
PART 1 — Monte Carlo Simulations    →  controlled bias–variance analysis (8 DGPs × 1000 replications)
PART 2 — Empirical Analysis         →  real-world validation on the Diabetes dataset (Efron et al., 2004)
```

**Central question:** How do statistical and machine learning variable selection methods behave under realistic violations of the Gaussian assumption — multicollinearity, outliers, structural breaks, and their combinations?

---

## Methods Compared

| Family | Method | Penalty | SAS Procedure |
|---|---|---|---|
| Statistical Learning | Forward Selection | None (p-value / AIC / SBC) | `PROC GLMSELECT` |
| Statistical Learning | Backward Elimination | None (p-value / AIC / SBC) | `PROC GLMSELECT` |
| Statistical Learning | Stepwise | None (AIC / SBC) | `PROC GLMSELECT` |
| Machine Learning | LASSO | L1 | `PROC GLMSELECT` |
| Machine Learning | LAR | None (progressive) | `PROC GLMSELECT` |
| Machine Learning | Elastic Net | L1 + L2 | `PROC GLMSELECT` |

---

## PART 1 — Monte Carlo Simulations

### Simulation Design

- **N =** 200 observations · **P =** 50 predictors · **MC =** 1,000 replications
- **True model:** 6 active variables `β = (1, −0.35, 0.15, 0.27, 0.57, −0.14)`, rest = 0
- **Seed:** 12345

### Performance Metrics

Each replication is classified into exactly one of four categories:

| Category | Definition |
|---|---|
| **Perfect Fit** | FP = 0 and FN = 0 — exactly the true support |
| **Overfit** | FP > 0 and FN = 0 — true vars selected + noise |
| **Underfit** | FP = 0 and FN > 0 — some true vars missed |
| **Wrong** | FP > 0 and FN > 0 — mix of missed + spurious |

> P(Perfect) + P(Overfit) + P(Underfit) + P(Wrong) = 1 by construction

### Bias–Variance Decomposition (Preliminary Simulation)

Before the DGP analysis, a Monte Carlo bias–variance illustration was run on `f(x) = 2x² + 1` with M = 1,000 simulations:

| Model | Squared Bias | Variance | Interpretation |
|---|---|---|---|
| Degree 0 (Underfit) | 0.2899 | 0.0062 | Stable but wrong |
| Degree 2 (Optimal) | 3.97e-6 | 0.0071 | Best tradeoff |
| Degree 8 (Overfit) | 3.72e-5 | 0.0210 | ~3× variance of optimal |

---

### DGP Results Summary

#### DGP 1 — Gaussian Benchmark (Independent, Normal)

| Method | Config | Perfect Fit | Overfit | Underfit | Wrong |
|---|---|---|---|---|---|
| Forward | Stop = SBC | ~36% | ~64% | 0% | 0% |
| Forward | Stop = SL(0.05) | ~11% | ~89% | 0% | 0% |
| Backward | Stop = SBC | 28.6% | 71.4% | 0% | 0% |
| Backward | Other stops | ~0% | ~100% | 0% | 0% |
| Stepwise | Any criterion | 38–40% | 60–62% | 0% | 0% |
| LASSO | Stop=SBC, Choose=CV | **33.40%** | ~66% | <1% | ~0% |
| LAR | Stop=SBC, Choose=CV | **31.80%** | ~65% | <1% | ~0% |
| Elastic Net | Stop=CV, Choose=CV | **30.50%** | ~69% | <1% | ~0% |
| Any method | Stop=AdjR² | <15% | >85% | 0% | 0% |

> **Key finding:** Stepwise is structurally more robust than Forward due to its add-drop mechanism. SBC is the only criterion that meaningfully controls overfitting. Underfitting is near-zero in a clean Gaussian setting.

---

#### DGP 2 — Multicollinearity (Toeplitz correlation on X1–X6, r = 0.8 to 0.6)

| Method | Config | Perfect Fit | Underfit | Notable |
|---|---|---|---|---|
| Forward | Stop = SBC | ~35% | 0% | Unaffected by correlation |
| Stepwise | Any criterion | ~35% | 0% | Identical to benchmark |
| LASSO | Stop=CV | <10% | ~50% | L1 selects 1 from correlated group |
| LAR | Stop=CV | <10% | ~50% | Same failure mode as LASSO |
| Elastic Net | Stop=CV, Choose=CV | **29.70%** | ~0% | L2 penalty recovers the group |
| Elastic Net | Choose=Cp | ~0% | **58%** | Cp fails under correlation |

> **Key finding:** Multicollinearity is the **critical vulnerability of L1 methods**. LAR and LASSO collapse (perfect fit <10%). Elastic Net's grouping effect is decisive here. Stepwise remains surprisingly stable.

---

#### DGP 3 — Outliers (10% contamination from N(4,1))

| Method | Config | Perfect Fit | Underfit | Notable |
|---|---|---|---|---|
| Forward / Stepwise | Any | ~38% | 0% | Unaffected — outliers don't confuse variable identity |
| Elastic Net | Stop=CV, Choose=CP | **63.10%** | ~0% | Best result across all DGPs |
| LAR | Stop=SBC, Choose=CV | **36.70%** | 9% | Better than Gaussian benchmark |
| LASSO | Stop=SBC, Choose=CV | **32.20%** | 9% | Conservative criteria improve fit |

> **Key finding:** Outliers paradoxically **improve** ML perfect fit rates — noise forces selection criteria to become more conservative. Elastic Net reaches its peak performance (63.1%) in this scenario.

---

#### DGP 4 — Structural Break (random tbreak ∈ [10%, 70%] of N, with β-shift)

| Method | Config | Perfect Fit | Wrong | Underfit | Notable |
|---|---|---|---|---|---|
| Forward | Any | <2% | 60–90% | — | Irreversible selection fails completely |
| Forward | Stop=SL | <2% | >80% | — | p-value tests uninformative under break |
| Backward | SBC | ~0% | ~75% | ~14% | Mix of FP and FN |
| Stepwise | SBC | ~2% | ~63% | ~34% | Add-drop reduces wrong rate slightly |
| LASSO | Stop=SBC | <1% | — | **88.10%** | SBC too strict — sees no signal |
| LAR | Stop=CV | <1% | **70–80%** | — | CV error non-representative under shift |
| Elastic Net | Stop=CV | <1% | **81.10%** | — | L2 penalty cannot reconcile regime change |

> **Key finding:** Structural break is an **insurmountable limitation** for all methods tested. No approach achieves meaningful perfect fit. Break renders both regularization and model selection criteria non-informative.

---

#### DGP 5 — Multicollinearity + Outliers (Iman-Conover method)

| Method | Config | Perfect Fit | Overfit | Underfit |
|---|---|---|---|---|
| Forward / Stepwise | Any | 35–36% | 64–65% | 0% |
| Elastic Net | Stop=CV, Choose=CV | **24.4%** | — | — |
| Elastic Net | Choose=Cp | ~0% | — | **58.5%** |
| LAR | Stop=SBC, Choose=CV | 17.7% | — | — |
| LASSO | Stop=SBC, Choose=CV | 17.3% | — | — |
| Any | Stop=AdjR² or AICC | <10% | >90% | — |

> **Key finding:** Combination of multicollinearity + outliers reduces all ML perfect fit rates. Elastic Net (CV) remains most stable. Mallows Cp is **systematically unreliable** under this DGP (58.5% underfit).

---

#### DGP 6 — Structural Break + Outliers

| Method | Config | Perfect Fit | Wrong | Underfit |
|---|---|---|---|---|
| Forward | Any | ~0.3% | >80% | ~12% |
| Stepwise | Any | ~1.5% | ~59% | ~38% |
| Elastic Net | Stop=SBC | ~0% | — | **91–97.6%** |
| Elastic Net | Stop=CV | ~0% | high | — |
| LAR | Stop=CV | ~0% | **70.9%** | — |
| LASSO | Stop=SBC | ~0% | — | **88.1%** |

> **Key finding:** The combination of break + outliers **neutralizes all methods**. Outliers falsify prediction error calculation for LAR; LASSO sees no signal at all; Elastic Net with SBC over-penalizes due to high variance from outliers.

---

#### DGP 7 — Multicollinearity + Structural Break

| Method | Config | Perfect Fit | Wrong | Underfit |
|---|---|---|---|---|
| Forward / Stepwise | Any | ~0% | dominant | dominant |
| Elastic Net | Stop=CV | ~0% | — | **87.9%** |
| Elastic Net | Stop=SBC | ~0% | 20–25% | **78%** |
| Elastic Net | Stop=AdjR² | ~0% | **~80%** | — |
| LASSO / LAR | Any | ~0% | — | — |

> **Key finding:** All methods fail when correlation and structural instability are combined. AdjR² should never be used in this context — it reaches 80% wrong model rate, failing to distinguish signal shifts from noise.

---

### Cross-DGP Summary

| DGP | Best method | Best Perfect Fit | Main failure |
|---|---|---|---|
| Gaussian | Stepwise (SBC) | 38–40% | Overfitting (~60%) |
| Multicollinearity | Elastic Net (CV) | 29.7% | L1 collapse (<10%) |
| Outliers | Elastic Net (CV/CP) | **63.1%** | Slight underfitting |
| Structural Break | None | <2% | Wrong model (>60%) |
| Corr + Outliers | Elastic Net (CV) | 24.4% | Cp failure (58.5% underfit) |
| Break + Outliers | None | <2% | Near-complete failure |
| Corr + Break | None | ~0% | Complete failure |

> **Overall conclusion:** SBC is the most reliable stopping criterion. Elastic Net is the most robust ML method. Structural breaks represent a fundamental limitation that no tested method can overcome.

---

## PART 2 — Empirical Analysis on the Diabetes Dataset

### Dataset

**Source:** Efron et al. (2004) — *Least Angle Regression*, Annals of Statistics

| Variable | Description |
|---|---|
| `AGE`, `SEX`, `BMI`, `BP` | Demographic & clinical |
| `S1` → `S6` | Blood serum measurements |
| `Y` | Disease progression score (continuous, 1 year after baseline) |

- **442 patients** · **10 predictors** · **1 continuous target**
- Split: **70% training / 30% validation**

### EDA Findings

- Strong multicollinearity detected: `S1–S2` correlation = **r = 0.90**
- Outliers present, notably for **BMI** and **S2** (extreme high values) and **S6** (both directions)
- → Dataset mirrors **DGP 5 (Multicollinearity + Outliers)** — Elastic Net expected to outperform

### Results

| Method | Selected Variables | # Vars | Validation-ASE |
|---|---|---|---|
| Stepwise (SBC) | SEX, BMI, BP, S3, S5 | 5 | 3146.28 |
| Stepwise (AIC) | SEX, BMI, BP, **S1**, S3, S5 | 6 | 3105.92 |
| LASSO (SBC + CV) | BMI, BP, S3, S5 | 4 | 2975.80 |
| **Elastic Net (SBC + CV)** | SEX, BMI, BP, **S2**, S3, S5 | 6 | **2816.91** |

### Interpretation

- **LASSO** (L1) selects only S3 among the correlated S2–S3 pair — too sparse, ASE = 2975
- **Elastic Net** (L1+L2) uses its grouping effect to retain **both S2 and S3**, reducing ASE to **2816** — a 5.4% improvement over LASSO
- **Stepwise** remains interpretable but is outperformed by regularization methods in predictive accuracy
- Confirms simulation finding from DGP 5: **Elastic Net dominates under multicollinearity + outliers**

---

## Repository Structure
```
├── bias var compromise.sas         # Monte Carlo bias–variance illustration (degrees 0, 2, 8)
├── Simulations and metrics.sas     # All 8 DGPs + FP/FN metric macro (1000 replications)
├── EMPIRICAL ANALYSIS.sas          # EDA (correlation matrix, boxplots) + Stepwise / LASSO / Elastic Net
├── Mémoire 1.pdf                   # Full research paper (theory, DGPs, results, conclusion)
└── README.md

---
```

## Tech Stack

![SAS](https://img.shields.io/badge/SAS-PROC%20GLMSELECT%20%7C%20PROC%20IML-blue?style=flat-square)
![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-1000%20replications-orange?style=flat-square)
![DGPs](https://img.shields.io/badge/DGPs-8%20scenarios-green?style=flat-square)
![CV](https://img.shields.io/badge/Validation-10--fold%20CV-purple?style=flat-square)

**Key SAS tools:** `PROC GLMSELECT`, `PROC IML`, `PROC CORR`, `PROC STDIZE`, `PROC SGPLOT`, `PROC SQL`, `%MACRO`

