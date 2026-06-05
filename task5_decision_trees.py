"""
Task 5: Decision Trees and Random Forests
AI & ML Internship - Heart Disease Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 0. Style
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3d',
    'grid.linewidth':   0.5,
    'font.family':      'DejaVu Sans',
})
ACCENT  = '#7c83fd'
GREEN   = '#56cfb2'
ORANGE  = '#f5a623'
RED     = '#f05454'
PALETTE = [ACCENT, GREEN, ORANGE, RED, '#c77dff', '#48cae4']

# ─────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  Task 5 — Decision Trees & Random Forests")
print("=" * 60)

df = pd.read_csv('heart_disease.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Target distribution:\n{df['target'].value_counts().rename({0:'No Disease', 1:'Disease'})}\n")

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs',
                 'restecg','thalach','exang','oldpeak','slope','ca','thal']
LABEL_MAP = {0: 'No Disease', 1: 'Disease'}

X = df[FEATURE_NAMES]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 2. Decision Tree (full depth — overfitting demo)
# ─────────────────────────────────────────────
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
acc_train_full = accuracy_score(y_train, dt_full.predict(X_train))
acc_test_full  = accuracy_score(y_test,  dt_full.predict(X_test))

print(f"\n── Decision Tree (no depth limit) ──────────────────")
print(f"   Train accuracy : {acc_train_full:.4f}")
print(f"   Test  accuracy : {acc_test_full:.4f}  ← likely overfit")

# ─────────────────────────────────────────────
# 3. Overfitting Analysis — vary max_depth
# ─────────────────────────────────────────────
depths       = range(1, 21)
train_scores = []
test_scores  = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test,   dt.predict(X_test)))

best_depth = int(np.argmax(test_scores)) + 1
print(f"\n── Depth Analysis ──────────────────────────────────")
print(f"   Best depth (test acc): {best_depth}  →  {max(test_scores):.4f}")

# ─────────────────────────────────────────────
# 4. Decision Tree (best depth)
# ─────────────────────────────────────────────
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_train, y_train)
y_pred_dt = dt_best.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n── Decision Tree (depth={best_depth}) ─────────────────────")
print(f"   Test accuracy : {acc_dt:.4f}")
print(classification_report(y_test, y_pred_dt,
                             target_names=['No Disease','Disease']))

# ─────────────────────────────────────────────
# 5. Random Forest
# ─────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"── Random Forest (100 trees) ───────────────────────")
print(f"   Test accuracy : {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf,
                             target_names=['No Disease','Disease']))

# ─────────────────────────────────────────────
# 6. Cross-Validation
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_dt = cross_val_score(dt_best, X, y, cv=cv, scoring='accuracy')
cv_rf = cross_val_score(rf,      X, y, cv=cv, scoring='accuracy')

print(f"\n── 5-Fold Cross-Validation ─────────────────────────")
print(f"   Decision Tree : {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")
print(f"   Random Forest : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

# ─────────────────────────────────────────────
# 7. Feature Importances
# ─────────────────────────────────────────────
fi_rf = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
fi_dt = pd.Series(dt_best.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

print(f"\n── Top-5 Feature Importances (Random Forest) ──────")
for feat, imp in fi_rf.head(5).items():
    print(f"   {feat:<12} {imp:.4f}")

# ═══════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════

# ── Figure 1: Overfitting Curve ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0f1117')
ax.plot(depths, train_scores, color=ACCENT,  lw=2.5, marker='o', ms=5, label='Train')
ax.plot(depths, test_scores,  color=GREEN,   lw=2.5, marker='s', ms=5, label='Test')
ax.axvline(best_depth, color=ORANGE, ls='--', lw=1.8, label=f'Best depth = {best_depth}')
ax.set_xlabel('Tree Depth', fontsize=13)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Overfitting Analysis — Decision Tree Depth vs Accuracy', fontsize=14, pad=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_overfitting_curve.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\n✅ Saved: fig1_overfitting_curve.png")

# ── Figure 2: Decision Tree Visualization ────────────────
fig, ax = plt.subplots(figsize=(20, 8))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')
plot_tree(dt_best,
          feature_names=FEATURE_NAMES,
          class_names=['No Disease', 'Disease'],
          filled=True,
          rounded=True,
          fontsize=9,
          ax=ax,
          impurity=True,
          proportion=False)
ax.set_title(f'Decision Tree (max_depth={best_depth})', fontsize=16, color='white', pad=14)
plt.tight_layout()
plt.savefig('fig2_decision_tree.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✅ Saved: fig2_decision_tree.png")

# ── Figure 3: Feature Importance Comparison ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f1117')
for ax, fi, title, color in zip(
        axes, [fi_rf, fi_dt],
        ['Random Forest Feature Importance', f'Decision Tree Feature Importance (depth={best_depth})'],
        [ACCENT, GREEN]):
    ax.set_facecolor('#1a1d27')
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=color, alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#cccccc')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, fi.max() * 1.25)
plt.suptitle('Feature Importances', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('fig3_feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✅ Saved: fig3_feature_importance.png")

# ── Figure 4: Confusion Matrices ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0f1117')
for ax, y_pred, title in zip(
        axes,
        [y_pred_dt, y_pred_rf],
        [f'Decision Tree (depth={best_depth})', 'Random Forest (100 trees)']):
    ax.set_facecolor('#1a1d27')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, pad=10)
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
plt.suptitle('Confusion Matrices', fontsize=15)
plt.tight_layout()
plt.savefig('fig4_confusion_matrices.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✅ Saved: fig4_confusion_matrices.png")

# ── Figure 5: Model Comparison Dashboard ─────────────────
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('#0f1117')

# Accuracy bar
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('#1a1d27')
models  = ['DT Full\n(no limit)', f'DT Best\n(depth={best_depth})', 'Random\nForest']
accs    = [acc_test_full, acc_dt, acc_rf]
colors  = [RED, ORANGE, GREEN]
bars = ax1.bar(models, accs, color=colors, width=0.5, edgecolor='none')
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=13, pad=10)
ax1.grid(axis='y', alpha=0.3)

# CV comparison
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('#1a1d27')
x = np.arange(5)
w = 0.35
ax2.bar(x - w/2, cv_dt, width=w, color=ORANGE, label=f'DT (mean={cv_dt.mean():.3f})')
ax2.bar(x + w/2, cv_rf, width=w, color=GREEN,  label=f'RF (mean={cv_rf.mean():.3f})')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('5-Fold Cross-Validation', fontsize=13, pad=10)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1.15)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Decision Tree vs Random Forest — Full Comparison',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('fig5_model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("✅ Saved: fig5_model_comparison.png")

print("\n" + "=" * 60)
print("  All figures saved successfully!")
print("=" * 60)
