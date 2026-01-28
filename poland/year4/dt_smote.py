# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:44:53 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  

# Seed
seed = 1
np.random.seed(seed)

# Dataset 
df = pd.read_csv('data(polish).csv')
df.columns = df.columns.str.replace(' ', '_')
df = df[df['year'] == 4].copy()

X = df.drop(columns=['class', 'year'])
y = df['class'].astype(int)

# Griglia iperparametri 
param_grid = {'clf__max_depth': [2, 3, 4, 5, 10, 15, 20, 30, 50]}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Placeholder per risultati OOF
metrics_per_fold = []
predictions = np.zeros(len(y), dtype=int)
probas = np.zeros(len(y), dtype=float)

# Modello Decision Tree (senza class_weight)
dt = DecisionTreeClassifier(
    random_state=seed
)

# Pipeline: imputazione -> scaling -> SMOTE -> DT
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),     
    ('scaler', StandardScaler()),                    
    ('smote', SMOTE(random_state=seed)),
    ('clf', dt)
])

# GridSearch sull’inner CV (solo training)
gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='recall',   # obiettivo: massimizzare recall 
    n_jobs=-1,
    refit=True
)

# Nested CV (outer loop)
for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    gs.fit(X_tr, y_tr)
    best_pipe = gs.best_estimator_

    # Predizioni sul test del fold esterno
    y_pred = best_pipe.predict(X_te)
    y_prob = best_pipe.predict_proba(X_te)[:, 1]

    predictions[test_idx] = y_pred
    probas[test_idx] = y_prob

    # Metriche per fold
    fpr_fold, tpr_fold, _ = roc_curve(y_te, y_prob)
    metrics_per_fold.append({
        "accuracy": accuracy_score(y_te, y_pred),
        "rec_pos": recall_score(y_te, y_pred),
        "prec_pos": precision_score(y_te, y_pred, zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average='macro'),
        "prec_macro": precision_score(y_te, y_pred, average='macro', zero_division=0),
        "rec_macro": recall_score(y_te, y_pred, average='macro', zero_division=0),
        "bal_acc": balanced_accuracy_score(y_te, y_pred),
        "auc": auc(fpr_fold, tpr_fold)
    })

# Salvataggio metriche + statistiche
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_dt_smote_year4_polish.csv", index=False)
metrics_mean = df_metrics.mean(numeric_only=True)
metrics_std  = df_metrics.std(numeric_only=True)

# Metriche OOF globali
report = classification_report(y, predictions, target_names=['Non-Bankrupt', 'Bankrupt'])
cm = confusion_matrix(y, predictions)

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione - Decision Tree (Year=4, SMOTE)")
plt.tight_layout()
plt.show()

# ROC OOF
fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Decision Tree (Year=4, SMOTE)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

