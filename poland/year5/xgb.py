# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:54:14 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
)

from xgboost import XGBClassifier

# -------------------------------
# 0) Seed
# -------------------------------
seed = 1
np.random.seed(seed)

# -------------------------------
# 1) Caricamento e preprocessing
# -------------------------------
df = pd.read_csv('american_bankruptcy.csv')
df.columns = df.columns.str.replace(' ', '_')
df.dropna(inplace=True)

# Rimuovi variabili non predittive
X = df.drop(columns=['status_label', 'company_name', 'year'])

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Target binario
y = df['status_label'].map({'failed': 1, 'alive': 0}).astype(int)

feature_names = X.columns.tolist()
n_features = len(feature_names)

# -------------------------------
# 2) Modello base + griglia iperparametri
# -------------------------------

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

# CV annidata
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# -------------------------------
# 3) Nested CV manuale
# -------------------------------
metrics_per_fold = []
predictions = np.zeros(len(y), dtype=int)     # OOF labels
probas = np.zeros(len(y), dtype=float)        # OOF probabilities
best_params_list = []
fi_matrix = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    counts = np.bincount(y_tr.to_numpy(), minlength=2)
    neg, pos = int(counts[0]), int(counts[1])
    spw = (neg / pos) if pos > 0 else 1.0

    xgb = XGBClassifier(
        random_state=seed,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=spw
    )

    gs = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='recall',
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)

    best_xgb = gs.best_estimator_
    best_xgb.fit(X_tr, y_tr)          
    best_params_list.append(gs.best_params_)

    y_pred = best_xgb.predict(X_te)
    y_prob = best_xgb.predict_proba(X_te)[:, 1]

    predictions[test_idx] = y_pred
    probas[test_idx] = y_prob

    # Metriche per fold
    fpr_fold, tpr_fold, _ = roc_curve(y_te, y_prob)
    fold_metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "rec_pos": recall_score(y_te, y_pred),
        "prec_pos": precision_score(y_te, y_pred, zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average='macro'),
        "prec_macro": precision_score(y_te, y_pred, average='macro', zero_division=0),
        "rec_macro": recall_score(y_te, y_pred, average='macro', zero_division=0),
        "bal_acc": balanced_accuracy_score(y_te, y_pred),
        "auc": auc(fpr_fold, tpr_fold)
    }
    # salva feature importance (gain)
    fi = best_xgb.feature_importances_
    if fi is not None and len(fi) == n_features:
        fi_matrix.append(fi)
    
    metrics_per_fold.append(fold_metrics)

# -------------------------------
# 4) Riassunto metriche
# -------------------------------
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_XG_american_kaggle.csv", index=False)
metrics_mean = df_metrics.mean()
metrics_std  = df_metrics.std()

# media feature importance
fi_mean = np.mean(np.vstack(fi_matrix), axis=0)
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": fi_mean
}).sort_values("importance_mean", ascending=False)

fi_df.to_csv("f_imp_mean_XGB_USA_kaggle.csv", index=False)


# -------------------------------
# 5) Report + Confusion matrix OOF
# -------------------------------
report = classification_report(y, predictions, target_names=['alive', 'failed'])
print("\nClassification report (OOF):\n", report)

cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['alive', 'failed'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione - XGBoost")
plt.tight_layout()
plt.show()

# -------------------------------
# 6) ROC OOF
# -------------------------------
fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC OOF = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - XGBoost')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# -------------------------------
# Barplot Top 20 Feature Importances (solo media)
# -------------------------------
top_k = 20
top_df = fi_df.head(top_k)[::-1]  # reverse per barh
plt.figure(figsize=(8, 10))
plt.barh(top_df["feature"], top_df["importance_mean"])
plt.xlabel("Importance (mean)")
plt.title(f"XGBoost USA Feature Importance — Top {top_k}")
plt.tight_layout()
plt.show()

