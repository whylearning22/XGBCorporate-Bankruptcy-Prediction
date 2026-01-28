# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:24:02 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

# Conta outlier per colonna
outlier_counts = (df[num_cols] > OUTLIER_THRESHOLD).sum()

# 1) Rimuovi colonne con troppi outlier
cols_to_drop = outlier_counts[outlier_counts > MAX_OUTLIERS_PER_COL].index.tolist()
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    num_cols = [c for c in num_cols if c not in cols_to_drop]

# 2) Sostituisci outlier con mediana
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


# -------------------------------
# 2) Modello + griglia iperparametri (Random Forest)
# -------------------------------

param_grid = { 
    'clf__n_estimators': [10, 20],
    'clf__max_depth': [2, 3, 4, 5, 10, 15, 20, 30, 50],
    'clf__max_features': [1, 2, 3, 4]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# -------------------------------
# 3) Nested CV 
# -------------------------------
metrics_per_fold = []
predictions = np.zeros_like(y)              # OOF labels
probas = np.zeros_like(y, dtype=float)      # OOF probabilities

for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=seed)),
        ('clf', RandomForestClassifier(random_state=seed))
    ])

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='recall',
        n_jobs=-1
    )
    
    gs.fit(X_tr, y_tr)

    best_pipe = gs.best_estimator_
    best_pipe.fit(X_tr, y_tr)
    
    # Predizioni sul test outer
    y_pred = best_pipe.predict(X_te)
    y_prob = best_pipe.predict_proba(X_te)[:, 1]

    # OOF save
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
    metrics_per_fold.append(fold_metrics)

# -------------------------------
# 4) Riassunto metriche
# -------------------------------
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_RF_smote_taiwan_out.csv", index=False)
metrics_mean = df_metrics.mean()
metrics_std  = df_metrics.std()
report = classification_report(y, predictions, target_names=['Non-Bankrupt', 'Bankrupt'])

# -------------------------------
# 5) Confusion matrix OOF
# -------------------------------
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione - Random Forest (Nested CV, SMOTE)")
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
plt.title('Curva ROC - Random Forest (Nested CV, SMOTE in PIPE)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
