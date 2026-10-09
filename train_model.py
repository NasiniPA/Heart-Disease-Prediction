"""
Heart Disease Prediction - Machine Learning Pipeline & Model Training
Course: Machine Learning Laboratory (Course Outcome: CO5)
Dataset: UCI Cleveland Heart Disease Dataset (N=303)
Run with: python train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure working directory is the script's folder in VS Code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR:
    os.chdir(BASE_DIR)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import joblib

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

print("=" * 70)
print("1. LOADING CLEVELAND DATASET")
print("=" * 70)
if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] dataset.csv not found at {DATASET_PATH}")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Missing values found: {df.isnull().sum().sum()}")
print(f"Target distribution (0=Healthy, 1=Heart Disease):\n{df['target'].value_counts()}")

# Preprocess & Split (80/20 Stratified)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n2. TRAIN-TEST SPLIT (80/20 Stratified): Train={len(X_train)}, Test={len(X_test)}")

# Define Standardized ML Pipelines
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Support Vector Machine': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ])
}

# 5-Fold Stratified Cross-Validation
print("\n" + "=" * 70)
print("3. 5-FOLD STRATIFIED CROSS-VALIDATION EVALUATION (CO5)")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"{name:24s} | Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit & Test Evaluation
test_res = {}
print("\n" + "=" * 70)
print("4. TEST SET PERFORMANCE MEASURES (N=61, Held-out)")
print("=" * 70)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    test_res[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm, 'prob': y_prob}
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f} | Recall (Sensitivity): {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:\n", cm)

# Save Figures if Matplotlib is installed
if HAS_PLT:
    print("\n" + "=" * 70)
    print("5. SAVING DIAGNOSTIC CHARTS TO plots/")
    print("=" * 70)

    try:
        # Figure 1: Class Distribution
        plt.figure(figsize=(6, 5))
        plt.bar(['No Disease (0)', 'Heart Disease (1)'], y.value_counts().values, color=['#64748b', '#e11d48'], width=0.5)
        plt.title("Figure 1: Target Class Distribution (UCI Cleveland)", fontsize=12)
        plt.ylabel("Patient Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "1_class_distribution.png"), dpi=300)
        plt.close()

        # Figure 2: Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Figure 2: Feature Correlation Heatmap with Target", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "2_correlation_heatmap.png"), dpi=300)
        plt.close()

        # Figure 3: Confusion Matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(test_res['Logistic Regression']['cm'], annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f"Figure 3: Confusion Matrix (Recall={test_res['Logistic Regression']['rec']:.2%})", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "3_confusion_matrix.png"), dpi=300)
        plt.close()

        # Figure 4: ROC Curves
        plt.figure(figsize=(7, 6))
        colors = {'Logistic Regression': '#2563eb', 'Random Forest': '#059669', 'Support Vector Machine': '#d97706'}
        for name, r in test_res.items():
            fpr, tpr, _ = roc_curve(y_test, r['prob'])
            plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", color=colors[name], lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance (AUC=0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("Figure 4: Receiver Operating Characteristic (ROC) Comparison", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "4_roc_curve.png"), dpi=300)
        plt.close()

        # Figure 5: Model Comparison
        plt.figure(figsize=(8, 5))
        metrics_keys = ['acc', 'prec', 'rec', 'f1', 'auc']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(labels))
        w = 0.25
        for i, (name, r) in enumerate(test_res.items()):
            plt.bar(x + (i - 1)*w, [r[k] for k in metrics_keys], width=w, label=name, color=list(colors.values())[i])
        plt.xticks(x, labels)
        plt.ylim(0.7, 1.0)
        plt.ylabel("Score")
        plt.title("Figure 5: Model Performance Comparison Bar Chart", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "5_model_comparison.png"), dpi=300)
        plt.close()
        print("Generated all 5 diagnostic figures successfully in ./plots")
    except Exception as e:
        print(f"Note: Plot rendering skipped or completed with warning: {e}")

# Save Model Pipeline
joblib.dump(models['Logistic Regression'], MODEL_PATH)
print(f"\n[SUCCESS] Saved trained production model pipeline to: '{MODEL_PATH}'")
