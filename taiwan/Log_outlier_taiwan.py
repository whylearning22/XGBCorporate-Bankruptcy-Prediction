# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 16:12:28 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

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
#   - soglia outlier: valori > 100 (rapporti finanziari implausibili)
#   - rimuovi colonne con >1000 outlier
#   - sostituisci i restanti outlier con la mediana della variabile
# ===========================================================
OUTLIER_THRESHOLD = 100
MAX_OUTLIERS_PER_COL = 1000
target_col = 'Bankrupt?'  # nome target nel dataset originale

# colonne numeriche escludendo il target (se è numerico)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)

# conta outlier per colonna
outlier_counts = (df[num_cols] > OUTLIER_THRESHOLD).sum()

# 1) colonne da rimuovere (troppe anomalie)
cols_to_drop = outlier_counts[outlier_counts > MAX_OUTLIERS_PER_COL].index.tolist()
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    # aggiorna lista colonne numeriche dopo il drop
    num_cols = [c for c in num_cols if c not in cols_to_drop]

# 2) sostituzione outlier con mediana (calcolata sui valori plausibili <= threshold)
for col in num_cols:
    mask_out = df[col] > OUTLIER_THRESHOLD
    if mask_out.any():
        # mediana robusta: solo sui valori non-outlier; fallback alla mediana globale se necessario
        if (df[col] <= OUTLIER_THRESHOLD).any():
            med = df.loc[df[col] <= OUTLIER_THRESHOLD, col].median()
        else:
            med = df[col].median()
        df.loc[mask_out, col] = med

# -------------------------------
# Feature/Target
# -------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)

# -------------------------------
# Modello + griglia iperparametri
# -------------------------------
lr = LogisticRegression(
    random_state=seed, class_weight='balanced', max_iter=1000
)
p_grid = {'C': [0.01, 0.1, 1, 10, 100]}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

clf = GridSearchCV(
    estimator=lr,
    param_grid=p_grid,
    cv=inner_cv,
    scoring='recall',
    n_jobs=-1
)

pipeline = Pipeline([
    ('transformer', StandardScaler()),
    ('estimator', clf)
])

# -------------------------------
# Nested CV manuale per metriche fold-by-fold
# -------------------------------
metrics_per_fold = []
predictions = np.zeros_like(y)  # predizioni OOF
probas = np.zeros_like(y, dtype=float)

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    predictions[test_idx] = y_pred
    probas[test_idx] = y_score

    # metriche per fold
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fold_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "rec_pos": recall_score(y_test, y_pred),
        "prec_pos": precision_score(y_test, y_pred, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "prec_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "rec_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "bal_acc": balanced_accuracy_score(y_test, y_pred),
        "auc": auc(fpr, tpr)
    }
    metrics_per_fold.append(fold_metrics)

# -------------------------------
# Media e deviazione standard
# -------------------------------
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_logistica_taiwan_outlier.csv", index=False)
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
plt.title("Matrice di Confusione - Nested CV outlier (OOF)")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Log Nested CV outlier (OOF)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
