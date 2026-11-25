# Insurance Charges: Prediction & the Causal Effect of Smoking

**How much does smoking actually cost in health insurance — after controlling for age, BMI and everything else?**

This project uses the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (1,338 policyholders) in two parts:

1. **Prediction** — a regression pipeline that predicts annual insurance charges.
2. **Causal inference** — estimating the causal effect of smoking on charges with three independent methods.

## Key results

| | Result |
|---|---|
| Prediction test RMSE | **≈ $8,330** (train ≈ $8,510, 10-fold CV ≈ $8,590 ± $850 — no overfitting) |
| Causal effect of smoking (S-Learner, CausalML) | **+$23,849 / year** (95% CI $22,722 – $24,975) |
| Causal effect of smoking (T-Learner XGBoost, CausalML) | **+$23,872 / year** (95% CI $23,281 – $24,463) |
| Causal effect of smoking (DoWhy, backdoor linear regression) | **+$23,849 / year** |

- All three methods agree: smoking adds roughly **$23,850 per year** in charges.
- **BMI drives ~92% of the variation in the smoking effect** — smokers with BMI above 30 face by far the highest costs.
- **Refutation tests pass:** adding a random common cause leaves the estimate unchanged; a placebo treatment drops it to about −$105.

## Methods

**Part 1 — Prediction**
- Stratified train/test split on smoking status (only ~20% of the sample smokes)
- EDA: skewed target, smoker × age and smoker × BMI interactions
- Outlier review with IQR and Isolation Forest (high BMIs kept — medically plausible)
- scikit-learn `Pipeline` + `ColumnTransformer` (median imputation, scaling, one-hot encoding)
- Linear regression on log(charges) via `TransformedTargetRegressor`, evaluated with 10-fold cross-validation

**Part 2 — Causal inference**
- Treatment: smoker · Outcome: charges · Controls: age, BMI, sex, children, region
- Meta-learners from **CausalML** (`LRSRegressor`, `XGBTRegressor`) with feature importance for effect heterogeneity
- **DoWhy** causal model with backdoor identification and refutation tests

## Limitations
- Observational data — unmeasured factors such as diet or exercise may bias the estimate.
- BMI is partly affected by smoking, so it is an imperfect control.
- The DoWhy estimate assumes a linear relationship despite the smoking × BMI interaction seen in the EDA.

## How to run

```bash
pip install -r requirements.txt
```

Download `insurance.csv` from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance), place it in this folder, then open `insurance_charges_causal_inference.ipynb`.

## Tools
Python · pandas · NumPy · Matplotlib · scikit-learn · CausalML · DoWhy · XGBoost

---
*Completed as part of the Master of Management in Analytics, McGill University.*
