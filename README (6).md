# Task 5: Decision Trees and Random Forests 🌳

**AI & ML Internship — Elevate Labs**

## 🎯 Objective
Learn tree-based models for classification using the Heart Disease Dataset.  
Tools: **Scikit-learn**, **Matplotlib**, **Seaborn**

---

## 📁 Files
| File | Description |
|---|---|
| `task5_decision_trees.py` | Main Python script (all steps) |
| `heart_disease.csv` | Heart Disease dataset (303 patients, 14 features) |
| `fig1_overfitting_curve.png` | Train vs Test accuracy across tree depths |
| `fig2_decision_tree.png` | Visualized Decision Tree |
| `fig3_feature_importance.png` | Feature importance — DT & RF |
| `fig4_confusion_matrices.png` | Confusion matrices for both models |
| `fig5_model_comparison.png` | Accuracy & cross-validation comparison |

---

## 📊 Dataset
**Heart Disease Dataset** — 303 patients, 13 features, binary target (0=No Disease, 1=Disease)

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1=male, 0=female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored |
| thal | Thalassemia type |

---

## 🔬 Steps Performed

### 1. Decision Tree — Full Depth (Overfitting)
- Trained with no depth limit → **Train accuracy: 100%**, **Test: ~83%**
- Clearly overfitting: memorizes training data

### 2. Overfitting Analysis — Controlling Tree Depth
- Swept depths 1–20, plotted train vs test accuracy
- Found best depth where test accuracy peaks

### 3. Decision Tree — Best Depth
- Retrained with optimal `max_depth`
- Evaluated with precision, recall, F1-score

### 4. Random Forest — 100 Trees (Ensemble)
- Bagging of 100 decision trees
- Compared accuracy against single tree

### 5. Feature Importance
- Extracted from both models
- Top features: `ca`, `age`, `cp`, `oldpeak`, `trestbps`

### 6. Cross-Validation (5-Fold)
| Model | CV Mean Accuracy | CV Std |
|---|---|---|
| Decision Tree (best depth) | ~0.84 | ~0.046 |
| Random Forest (100 trees) | ~0.87 | ~0.017 |

> Random Forest has **higher accuracy AND lower variance** — more robust!

---

## 📈 Results Summary
| Model | Test Accuracy |
|---|---|
| Decision Tree (no limit) | ~0.84 — overfits |
| Decision Tree (best depth) | ~0.89 |
| Random Forest (100 trees) | ~0.87 |

---

## 💡 Interview Q&A

**1. How does a decision tree work?**  
It recursively splits data by the feature+threshold that best separates classes, using metrics like Gini impurity or entropy. Each leaf node represents a predicted class.

**2. What is entropy and information gain?**  
Entropy measures disorder/impurity in a node: `H = -Σ p·log₂(p)`. Information gain is the reduction in entropy after a split — the tree picks the split with the highest gain.

**3. How is Random Forest better than a single tree?**  
RF trains many trees on random data subsets (bagging) and random feature subsets, then averages predictions. This reduces variance and overfitting while improving generalization.

**4. What is overfitting and how do you prevent it?**  
Overfitting = model memorizes training data but fails on new data (high train acc, low test acc). Prevention: limit `max_depth`, set `min_samples_split`, use pruning, or switch to ensemble methods.

**5. What is bagging?**  
Bootstrap Aggregating — train multiple models on different random samples (with replacement) of the training data, then aggregate their predictions. Reduces variance without increasing bias.

**6. How do you visualize a decision tree?**  
Using `sklearn.tree.plot_tree()` or `export_graphviz()` + Graphviz. Each node shows the split condition, impurity, sample count, and class distribution.

**7. How do you interpret feature importance?**  
Feature importance = total weighted impurity reduction a feature provides across all splits. Higher = more influential. Accessible via `model.feature_importances_`.

**8. Pros/cons of Random Forests?**  
✅ High accuracy, handles missing values, robust to outliers, provides feature importance  
❌ Slow to predict on large datasets, less interpretable than single trees, memory-intensive

---

## 🛠️ Setup & Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn

python task5_decision_trees.py
```

---

*Submitted by [@BhavyaMathur1](https://github.com/BhavyaMathur1) as part of AI & ML Internship — Task 5*
