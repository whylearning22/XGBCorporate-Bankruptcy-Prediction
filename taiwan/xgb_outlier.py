# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:45:14 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

from xgboost import XGBClassifier

# -------------------------------
# Seed
# -------------------------------
seed = 1
np.random.seed(seed)

# -------------------------------
# Dataset
# -------------------------------
df = pd.read_csv('data (TAIWAN).csv')
df.columns = df.columns.str.replace(' ', '_')
df.dropna(inplace=True)

# ===========================================================
# Gestione OUTLIER "stile Brenes"
# ===========================================================
OUTLIER_THRESHOLD = 100
MAX_OUTLIERS_PER_COL = 1000
target_col = 'Bankrupt?'

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)

# conta outlier per colonna
outlier_counts = (df[num_cols] > OUTLIER_THRESHOLD).sum()

# 1) rimuovi colonne con troppi outlier
cols_to_drop = outlier_counts[outlier_counts > MAX_OUTLIERS_PER_COL].index.tolist()
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    num_cols = [c for c in num_cols if c not in cols_to_drop]

# 2) sostituisci outlier con mediana (valori <= soglia)
for col in num_cols:
    mask_out = df[col] > OUTLIER_THRESHOLD
    if mask_out.any():
        med = df.loc[df[col] <= OUTLIER_THRESHOLD, col].median()
        df.loc[mask_out, col] = med

# -------------------------------
# Feature / Target
# -------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)
feature_names = X.columns.tolist()
n_features = len(feature_names)


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)



# -------------------------------
# Nested CV manuale per metriche fold-by-fold
# -------------------------------
metrics_per_fold = []
predictions = np.zeros_like(y)  # per salvare predizioni OOF
probas = np.zeros_like(y, dtype=float)
fi_matrix = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Calcolo scale_pos_weight SOLO sul training esterno
    counts = np.bincount(y_train.to_numpy(), minlength=2)
    neg, pos = int(counts[0]), int(counts[1])
    spw = (neg / pos) if pos > 0 else 1.0

    # Modello con spw per questo fold
    xgb = XGBClassifier(
        random_state=seed,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=spw
    )
    
    clf = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='recall',
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    best_est = clf.best_estimator_
    
    y_pred = best_est.predict(X_test)
    y_score = best_est.predict_proba(X_test)[:, 1]

    # salvataggio OOF
    predictions[test_idx] = y_pred
    probas[test_idx] = y_score

    # metriche fold
    fpr_fold, tpr_fold, _ = roc_curve(y_test, y_score)
    fold_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "rec_pos": recall_score(y_test, y_pred),
        "prec_pos": precision_score(y_test, y_pred, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "prec_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "rec_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "bal_acc": balanced_accuracy_score(y_test, y_pred),
        "auc": auc(fpr_fold, tpr_fold)
    }
    # salva feature importance (gain)
    fi = best_est.feature_importances_
    if fi is not None and len(fi) == n_features:
        fi_matrix.append(fi)
    
    metrics_per_fold.append(fold_metrics)

# -------------------------------
# Media e deviazione standard
# -------------------------------
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_XG_taiwan.csv", index=False)
metrics_mean = df_metrics.mean()
metrics_std  = df_metrics.std()

# media feature importance
fi_mean = np.mean(np.vstack(fi_matrix), axis=0)
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": fi_mean
}).sort_values("importance_mean", ascending=False)

fi_df.to_csv("xgb_feature_importances_mean_out.csv", index=False)
print("\nTop 20 Feature Importances (media sui fold):")
print(fi_df.head(20))

# -------------------------------
# Metriche globali OOF
# -------------------------------
report = classification_report(y, predictions, target_names=['Non-Bankrupt', 'Bankrupt'])
cm = confusion_matrix(y, predictions)

# -------------------------------
# Grafici
# -------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
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
plt.title(f"XGBoost Feature Importance — Top {top_k} (Media Nested CV)")
plt.tight_layout()
plt.show()

print("\nUltimo best params (CV interna):", clf.best_params_)