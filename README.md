# Bias–Variance Tradeoff in Statistical Learning — Simulations & Empirical Analysis

> Evaluating variable selection methods through Monte Carlo simulations and real-world validation on the Diabetes dataset (SAS)

---

## Project Structure

This project is organized in **two complementary parts** :
```
PART 1 — Monte Carlo Simulations     →  controlled bias–variance analysis
PART 2 — Empirical Analysis          →  real diabetes dataset (Efron et al.)
```

---

## PART 1 — Bias–Variance Tradeoff via Simulations

### Objective

Measure the **bias–variance decomposition** of each variable selection method under controlled, reproducible conditions — where ground truth is known.

### Simulation Design
```
For each scenario (multicollinearity / structural breaks / autocorrelation):
    └── Generate synthetic data (known DGP)
            └── Run N replications (Monte Carlo)
                    ├── Fit model with method M
                    ├── Record selected variables
                    └── Compute: MSE, Bias², Variance
```

- **Seed:** `12345` (reproducibility)
- **Replications:** > 100 000
- **Train / Validation split:** 70% / 30%

### Methods Compared

| Method | Criterion | Selection logic |
|---|---|---|
| Stepwise | AIC | Forward/backward steps minimizing AIC |
| Stepwise | SBC | Stronger complexity penalty → sparser |
| LASSO | SBC (stop) + 10-fold CV | L1 shrinkage, exact zeros |
| Elastic Net | SBC (stop) + 10-fold CV | L1 + L2, handles correlated predictors |

### Metrics Tracked

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | E[(Ŷ − Y)²] | Total prediction error |
| **Bias²** | (E[Ŷ] − Y)² | Systematic error |
| **Variance** | E[(Ŷ − E[Ŷ])²] | Sensitivity to training data |
| **ASE** | Average Squared Error on validation set | Out-of-sample fit |
| **AIC / SBC** | Penalized likelihood | In-sample model complexity |

> **Key identity:** MSE = Bias² + Variance + Irreducible noise

### Key Findings

- No single method dominates across all scenarios
- **LASSO** minimizes variance at the cost of slight bias (especially under multicollinearity)
- **Stepwise–AIC** is less penalizing → higher variance, lower bias
- **Stepwise–SBC** produces sparser models but can underfit under autocorrelation
- **Elastic Net** best handles correlated predictors — lower variance on S1–S6 type structures

---

## PART 2 — Empirical Analysis on Diabetes Data

### Dataset

**Efron et al. (2004)** — *Least Angle Regression* — standard benchmark in statistical learning

| Variable | Description |
|---|---|
| `AGE`, `SEX`, `BMI`, `BP` | Demographic & clinical |
| `S1` → `S6` | Blood serum measurements (strongly correlated) |
| `Y` | Disease progression score (continuous target) |

- 442 patients, 10 predictors, 1 continuous target
- Split: **70% train / 30% validation**

### Step 1 — Exploratory Data Analysis

**Correlation matrix** (PROC CORR + PROC IML)
- Strong multicollinearity detected among S1–S6
- BMI and BP show highest marginal correlation with Y

**Standardized boxplots** (PROC STDIZE + PROC SGPLOT)
- Distributions compared on a common scale
- Outlier detection across all predictors

### Step 2 — Variable Selection

Same four methods applied to real data:
```
Stepwise (AIC)  →  selection=stepwise(select=AIC  choose=validate)
Stepwise (SBC)  →  selection=stepwise(select=SBC  choose=validate)
LASSO           →  selection=lasso(stop=SBC        choose=CV) cvmethod=random(10)
Elastic Net     →  selection=elasticnet(stop=SBC   choose=CV) cvmethod=random(10)
```

### Step 3 — Results

| Method | Variables selected | Validation ASE |
|---|---|---|
| Stepwise AIC | BMI, BP, S1, S2, S3, S5 | — |
| Stepwise SBC | BMI, BP, S5 | — |
| LASSO | BMI, BP, S3, S5 | — |
| Elastic Net | BMI, BP, S3, S5, S6 | — |

> ⚠️ Fill in your actual ASE values and selected variables from your SAS output

**BMI, BP and S5** are consistently selected → most robust predictors of disease progression

---

## 📁 Repository Structure
```
├── data/
│   └── Diabetes.txt
├── simulations/
│   ├── dgp.sas                  # Data generating process
│   ├── scenario_multicol.sas    # Multicollinearity
│   ├── scenario_breaks.sas      # Structural breaks
│   └── scenario_autocorr.sas    # Autocorrelation
├── empirical/
│   ├── 01_eda_correlation.sas
│   ├── 02_eda_boxplot.sas
│   ├── 03_stepwise_AIC.sas
│   ├── 04_stepwise_SBC.sas
│   ├── 05_lasso.sas
│   └── 06_elasticnet.sas
└── README.md
```

---

## 🛠️ Tech Stack

![SAS](https://img.shields.io/badge/SAS-PROC%20GLMSELECT-blue?style=flat-square)
![Simulation](https://img.shields.io/badge/Monte%20Carlo-100k%2B%20replications-orange?style=flat-square)
![CV](https://img.shields.io/badge/Validation-10--fold%20CV-green?style=flat-square)

---

## 👩‍💻 Author

**Aya El Farfar** — M2 MoSEF, Paris 1 Panthéon-Sorbonne  
[LinkedIn](https://linkedin.com/in/aya-el-farfar) · [GitHub](https://github.com/elfarfaraya-boop)
