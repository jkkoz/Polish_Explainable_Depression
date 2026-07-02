import pandas as pd
from pathlib import Path
import json
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ===== config =====
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10 

df_train = pd.read_csv("feature_train.csv")
df_val = pd.read_csv("feature_val.csv")
df_test = pd.read_csv("feature_test.csv")

senti = ('positive', 'neutral', 'negative')
senti_map = (1, 0, -1)
mapping = dict(zip(senti, senti_map))

df_train['sentiment'] = df_train['sentiment'].map(mapping)
df_val['sentiment'] = df_val['sentiment'].map(mapping)
df_test['sentiment'] = df_test['sentiment'].map(mapping)
features = ['n_1st_pronoun','n_sing', 'rate_past',
            'rate_sing', 'sentiment','rate_cond', 'n_cond',
            'n_1st_verb', 'emoji', 'n_words', 'n_past','rate_verb','rate_pron',
            'absolutist',
            ]

df_train = df_train.dropna(subset=features)
df_val = df_val.dropna(subset=features)
df_test = df_test.dropna(subset=features)


X_train = df_train[features].values
Y_train = df_train['label'].values
X_val = df_val[features].values
Y_val = df_val['label'].values
X_test = df_test[features].values
Y_test = df_test['label'].values


# standardize features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test  = scaler.transform(X_test) 

# ===== helper to evaluate =====
def evaluate_model(name, model, X, y_true, proba=None):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro")
    prec_weighted = precision_score(y_true, y_pred, average="weighted")
    rec_macro = recall_score(y_true, y_pred, average="macro")
    rec_weighted = recall_score(y_true, y_pred, average="weighted")

    auc = None
    if proba is not None:
        auc = roc_auc_score(y_true, proba[:, 1])
    print(f"\n=== {name} ===")
    print(f"accuracy: {acc:.4f}")
    print(f"precision_macro: {prec_macro:.4f}  precision_weighted: {prec_weighted:.4f}")
    print(f"recall_macro: {rec_macro:.4f}  recall_weighted: {rec_weighted:.4f}")
    print(f"f1_macro: {f1_macro:.4f}  f1_weighted: {f1_weighted:.4f}")
    if auc is not None:
        print(f"roc_auc: {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(auc) if auc is not None else None,
    }

def save_if_not_exists(obj, path):
    path = Path(path)
    if not path.exists():
        joblib.dump(obj, path)
        print(f"Saved: {path}")
    else:
        print(f"Already exists: {path}, skipping save")

def run_cv(name, estimator, X, y, cv, scoring):
    """Run cross_validate and print mean ± std for every metric."""
    results = cross_validate(
        estimator, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    print(f"\n=== 5-Fold CV: {name} ===")
    summary = {}
    for key, scores in results.items():
        if key.startswith("test_"):
            metric = key[5:]   # strip "test_" prefix
            mean, std = scores.mean(), scores.std()
            print(f"  {metric:<25} {mean:.4f} ± {std:.4f}")
            summary[metric] = {"mean": float(mean), "std": float(std)}
    return summary

# ===== CV setup =====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
 
scoring = {
    "accuracy":        "accuracy",
    "f1_macro":        "f1_macro",
    "f1_weighted":     "f1_weighted",
    "precision_macro": "precision_macro",
    "recall_macro":    "recall_macro",
    "roc_auc":         "roc_auc",   
}

# ===== Logistic Regression =====
logreg = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

cv_logreg = run_cv("Logistic Regression", logreg, X_train, Y_train, cv, scoring)

logreg.fit(X_train, Y_train)

# save logistic regression
# save_if_not_exists(logreg, "baselines/inMethods/logreg.joblib")

val_proba_lr = logreg.predict_proba(X_val)
test_proba_lr = logreg.predict_proba(X_test)

metrics_val_lr  = evaluate_model("Logistic Regression validation", logreg, X_val, Y_val, proba=val_proba_lr)
metrics_test_lr = evaluate_model("Logistic Regression test", logreg, X_test, Y_test, proba=test_proba_lr)

# ===== Linear SVM with probability calibration with sigmoid =====

svm_linear = LinearSVC(C=1.0, random_state=RANDOM_STATE)
svm = CalibratedClassifierCV(svm_linear, method="sigmoid", cv=5)

cv_svm = run_cv("Linear SVM (calibrated)", svm, X_train, Y_train, cv, scoring)

svm.fit(X_train, Y_train)

# save svm
# save_if_not_exists(svm, "baselines/inMethods/linear_svm_calibrated.joblib")

val_proba_svm = svm.predict_proba(X_val)
test_proba_svm = svm.predict_proba(X_test)
svm_for_pred = svm

metrics_val_svm  = evaluate_model("Linear SVM validation", svm_for_pred, X_val, Y_val, proba=val_proba_svm)
metrics_test_svm = evaluate_model("Linear SVM test", svm_for_pred, X_test, Y_test, proba=test_proba_svm)

metrics = {
    "cv_logreg":      cv_logreg,
    "cv_linear_svm":  cv_svm,
    "val_logreg": metrics_val_lr,
    "test_logreg": metrics_test_lr,
    "val_linear_svm": metrics_val_svm,
    "test_linear_svm": metrics_test_svm,
}

# save metrics
# with open("baselines/inMethods/metrics.json", "w") as f:
#     json.dump(metrics, f, indent=2)

summary_rows = []
for split in ["val", "test"]:
    for model_name, m in [("logreg", metrics[f"{split}_logreg"]), ("linear_svm", metrics[f"{split}_linear_svm"])]:
        summary_rows.append({
            "split": split,
            "model": model_name,
            **m
        })


summary = pd.DataFrame(summary_rows)

# save summary
# summary.to_csv("baselines/inMethods/metrics_summary.csv", index=False)

print(summary)

# Coefficients and Odds Ratios
coefficients = logreg.coef_[0]
odds_ratios = np.exp(coefficients)

# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': df_train[features].columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})
print("\nFeature Importance (Coefficient and Odds Ratio):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))

perm_importance = permutation_importance(logreg, X_test, Y_test, n_repeats=30, random_state=42, n_jobs=-1)
perm_importance_df = pd.DataFrame({
    'Feature': df_train[features].columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
})
print("\nPermutation Importance:")
print(perm_importance_df.sort_values(by='Importance Mean', ascending=False))
