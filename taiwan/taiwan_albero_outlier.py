# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 16:30:35 2025

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
from sklearn.tree import DecisionTreeClassifier

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

# colonne numeriche (escludi target)
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

# 2) sostituisci outlier con mediana
for col in num_cols:
    mask_out = df[col] > OUTLIER_THRESHOLD
    if mask_out.any():
        med = df.loc[df[col] <= OUTLIER_THRESHOLD, col].median()
        df.loc[mask_out, col] = med

# -------------------------------
# Feature e target
# -------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)

# -------------------------------
# Modello Decision Tree + griglia iperparametri
# -------------------------------
dt = DecisionTreeClassifier(
    random_state=seed,
    class_weight='balanced'
)

# griglia richiesta
p_grid = {'max_depth': [2, 3, 4, 5, 10, 15, 20, 30, 50]}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


clf = GridSearchCV(
    estimator=dt,
    param_grid=p_grid,
    cv=inner_cv,
    scoring='recall',
    n_jobs=-1
)

# -------------------------------
# Nested CV manuale per metriche fold-by-fold
# -------------------------------
metrics_per_fold = []
predictions = np.zeros_like(y)
probas = np.zeros_like(y, dtype=float)

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf.fit(X_train, y_train)

    best_clf= clf.best_estimator_
    best_clf.fit(X_train, y_train)  

    # Predizioni sul test outer
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)[:, 1]

    # Salva predizioni OOF
    predictions[test_idx] = y_pred
    probas[test_idx] = y_prob

    fpr_fold, tpr_fold, _ = roc_curve(y_test, y_prob)
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
    metrics_per_fold.append(fold_metrics)

# -------------------------------
# Media e deviazione standard metriche
# -------------------------------
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_DT_taiwan_outlier.csv", index=False)
metrics_mean = df_metrics.mean()
metrics_std  = df_metrics.std()

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
plt.title("Matrice di Confusione - Decision Tree (OOF)")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Decision Tree Nested CV (OOF)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
