# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 16:11:55 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
)

# -------------------------------
# 0. Seed
# -------------------------------
seed = 1
np.random.seed(seed)

# -------------------------------
# 1. Caricamento e preprocessing
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

# -------------------------------
# 2. Modello + griglia iperparametri (RF)
# -------------------------------
rf_base = RandomForestClassifier(
    random_state=seed,
    class_weight='balanced',
    n_jobs=-1,
)

param_grid = { 
    'n_estimators': [10, 20],
    'max_depth': [2, 3, 4, 5, 10, 15, 20, 30, 50],
    'max_features': [1, 2, 3, 4]   
}

# CV annidata
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# -------------------------------
# 3. Nested CV manuale
# -------------------------------
metrics_per_fold = []
predictions = np.zeros_like(y)              # OOF labels
probas = np.zeros_like(y, dtype=float)      # OOF probabilities
best_params_list = []                       # per salvare i best param per fold

for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # Tuning interno
    gs = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='recall',
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)

    best_rf = gs.best_estimator_
    best_params_list.append(gs.best_params_)

    # Predizioni sul test esterno
    y_pred = best_rf.predict(X_te)
    y_prob = best_rf.predict_proba(X_te)[:, 1]

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
# 4. Riassunto metriche
# -------------------------------
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_RF_american_kaggle.csv", index=False)
metrics_mean = df_metrics.mean()
metrics_std  = df_metrics.std()

# Report globale OOF
report = classification_report(y, predictions, target_names=['alive', 'failed'])

# -------------------------------
# 5. Confusion matrix OOF
# -------------------------------
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['alive', 'failed'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione - Random Forest")
plt.tight_layout()
plt.show()

# -------------------------------
# 6. ROC OOF
# -------------------------------
fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC OOF = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Random Forest')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


