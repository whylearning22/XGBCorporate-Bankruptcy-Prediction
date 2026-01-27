# XGBCorporate-Bankruptcy-Prediction
This repository provides the code to reproduce the experiments presented in the paper:

**“Generalizable Machine Learning for Corporate Bankruptcy Prediction: An XGBoost-Based Framework”**

The proposed framework is based on **XGBoost** and is designed for **highly imbalanced bankruptcy prediction**. The evaluation is conducted on three benchmark datasets (Taiwan, USA, Poland) under a **leakage-resistant pipeline** with **nested cross-validation** and consistent preprocessing.

---

## Key Features

- **Unified evaluation pipeline** across heterogeneous datasets
- **Imbalance handling** via:
  - XGBoost `scale_pos_weight`
  - Optional **SMOTE** applied **only on training folds**
- **Nested Cross-Validation**:
  - Outer loop: model assessment
  - Inner loop: hyperparameter tuning (grid search)
- Baselines included: **Logistic Regression (LR)**, **Decision Tree (DT)**, **Random Forest (RF)**
- Metrics: AUC, Recall, Accuracy, Macro-Recall, Macro-Precision, Macro-F1, Confusion Matrix
- **Interpretability**: global feature importance from XGBoost (split frequency)
