# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 00:07:29 2025

@author: Graziano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from xgboost import XGBClassifier

# Seed
seed = 1
np.random.seed(seed)

# Dataset 
df = pd.read_csv('data(polish).csv')
df.columns = df.columns.str.replace(' ', '_')
df = df[df['year'] == 2].copy()

X = df.drop(columns=['class', 'year'])
y = df['class'].astype(int)

# Griglia iperparametri 
param_grid = {
    'clf__n_estimators': [100, 300],
    'clf__max_depth': [3, 5, 10],
    'clf__learning_rate': [0.01, 0.1]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

metrics_per_fold = []
predictions = np.zeros(len(y), dtype=int)
probas = np.zeros(len(y), dtype=float)

# Lista per le feature importance
feature_importances = []

# Nested CV (outer loop)
for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # --- Calcolo bilanciamento SOLO sul training 
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    # --- Modello XGB con scale_pos_weight
    xgb = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist'  
    )

    # --- Pipeline ---
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', xgb)
    ])

    # --- Grid Search sull’inner CV 
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='recall',   
        n_jobs=-1,
        refit=True
    )

    gs.fit(X_tr, y_tr)
    best_pipe = gs.best_estimator_

    # --- Predizioni sul test del fold esterno
    y_pred = best_pipe.predict(X_te)
    y_prob = best_pipe.predict_proba(X_te)[:, 1]

    predictions[test_idx] = y_pred
    probas[test_idx] = y_prob

    # --- Metriche per fold
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

 # --- Feature importance del modello del fold corrente
    fold_importance = best_pipe['clf'].feature_importances_
    feature_importances.append(fold_importance)

# --- Media delle feature importance sui 10 fold
mean_importances = np.mean(feature_importances, axis=0)
df_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Mean': mean_importances
}).sort_values(by='Importance_Mean', ascending=False)

# --- Salvataggio metriche + statistiche
df_metrics = pd.DataFrame(metrics_per_fold)
df_metrics.to_csv("metriche_xgboost_year2_polish.csv", index=False)
metrics_mean = df_metrics.mean(numeric_only=True)
metrics_std  = df_metrics.std(numeric_only=True)

# --- Top-N (max 20 o meno se le feature sono < 20)
top_n = min(20, len(df_importances))
df_importances_top = df_importances.head(top_n)
df_importances_top.to_csv("feature_importance_top20_xgboost_year2_polish.csv", index=False)

# --- Plot feature importance Top-N
plt.figure(figsize=(10, 6))
plt.barh(df_importances_top['Feature'], df_importances_top['Importance_Mean'])
plt.xlabel("Mean Feature Importance")
plt.ylabel("Feature")
plt.title(f"Top {top_n} XGBoost Feature Importances (Year=2)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- Metriche OOF globali
report = classification_report(y, predictions, target_names=['Non-Bankrupt', 'Bankrupt'])
cm = confusion_matrix(y, predictions)

# --- Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title("Matrice di Confusione - XGBoost (Year=2)")
plt.tight_layout()
plt.show()

# --- ROC OOF
fpr, tpr, _ = roc_curve(y, probas)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - XGBoost (Year=2)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

