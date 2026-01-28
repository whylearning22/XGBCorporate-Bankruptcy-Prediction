# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 20:31:01 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------------
# 0) Seed
# -------------------------------
seed = 1
np.random.seed(seed)

# -------------------------------
# 1) Dataset + preprocessing (american)
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
# 2) Modello + griglia iperparametri (XGBoost)
# -------------------------------
xgb_inner = XGBClassifier(
    random_state=seed,
    n_jobs=-1,
    eval_metric='logloss'
)

param_grid = { 
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [2, 3, 4, 5, 10],
    'clf__learning_rate': [0.01, 0.1, 0.2]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# -------------------------------
# 3) Nested CV con SMOTE solo nell'outer
# -------------------------------
metrics_per_fold = []
predictions = np.zeros_like(y)              # OOF labels
probas = np.zeros_like(y, dtype=float)      # OOF probabilities

for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=seed)),
        ('clf', xgb_inner)
    ])

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='recall',
        n_jobs=-1
    )
    #un ciclo di valutazione di tutti i modelli (con combinazioni di iperparametri),
    #ciascuno con SMOTE applicato solo sui fold di training interni
    gs.fit(X_tr, y_tr)  # Qui SMOTE viene applicato solo nel training fold di ogni split

    #prende il miglior modello trovato (best_pipe, ovvero il Pipeline con SMOTE + XGBoost configurato con i best params),
    #e lo riaddestra da zero su tutto X_tr, cioè su tutto il training dell’outer fold.
    best_pipe = gs.best_estimator_
    best_pipe.fit(X_tr, y_tr)  # Qui SMOTE è applicato solo su training fold (non serve riapplicarlo)
    
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
# Salva solo le metriche dei fold su CSV
df_metrics.to_csv("metriche_XG_smote_american_kaggle.csv", index=False)

metrics_mean = df_metrics.mean()
metrics_std  = df_metrics.std()
report = classification_report(y, predictions, target_names=['alive', 'failed'])

# -------------------------------
# 5) Confusion matrix OOF
# -------------------------------
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['alive', 'failed'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione - XGBoost (SMOTE)")
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
plt.title("Curva ROC - XGBoost (SMOTE)")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


