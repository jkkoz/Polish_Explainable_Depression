# Code adapted from https://github.com/sajjadIslam2619/mental-health-disorder-analysis.git

import pandas as pd
from pathlib import Path
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import joblib

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


df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer('sdadas/st-polish-paraphrase-from-distilroberta', device=device) # or SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

def embed_texts(model, texts, batch_size=256, show_progress=True):
    texts = ["" if pd.isna(t) else str(t) for t in texts]

    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=show_progress,    
    )

X_train = embed_texts(st_model, df_train["clean_text"])
y_train = df_train["label"].to_numpy()

X_val   = embed_texts(st_model, df_val["clean_text"])
y_val   = df_val["label"].to_numpy()

X_test  = embed_texts(st_model, df_test["clean_text"])
y_test  = df_test["label"].to_numpy()

print("Embedding shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Embeddings computed on device:", device)

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
        print(f"roc_auc_ovr: {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_ovr": float(auc) if auc is not None else None,
    }

def save_if_not_exists(obj, path):
    path = Path(path)
    if not path.exists():
        joblib.dump(obj, path)
        print(f"Saved: {path}")
    else:
        print(f"Already exists: {path}, skipping save")

# ===== Logistic Regression =====
logreg = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


logreg.fit(X_train, y_train)
save_if_not_exists(logreg, "test1/logreg.joblib")

val_proba_lr = logreg.predict_proba(X_val)
test_proba_lr = logreg.predict_proba(X_test)

metrics_val_lr  = evaluate_model("Logistic Regression validation", logreg, X_val, y_val, proba=val_proba_lr)
metrics_test_lr = evaluate_model("Logistic Regression test", logreg, X_test, y_test, proba=test_proba_lr)

# ===== Linear SVM with probability calibration with sigmoid =====

svm_linear = LinearSVC(C=1.0, random_state=RANDOM_STATE)
svm = CalibratedClassifierCV(svm_linear, method="sigmoid", cv=5)
svm.fit(X_train, y_train)
save_if_not_exists(svm, "test1/linear_svm_calibrated.joblib")

val_proba_svm = svm.predict_proba(X_val)
test_proba_svm = svm.predict_proba(X_test)
svm_for_pred = svm

metrics_val_svm  = evaluate_model("Linear SVM validation", svm_for_pred, X_val, y_val, proba=val_proba_svm)
metrics_test_svm = evaluate_model("Linear SVM test", svm_for_pred, X_test, y_test, proba=test_proba_svm)

# save models
# joblib.dump(logreg,"test1/logreg.joblib")
# joblib.dump(svm_for_pred, "test1/linear_svm_calibrated.joblib")

metrics = {
    "val_logreg": metrics_val_lr,
    "test_logreg": metrics_test_lr,
    "val_linear_svm": metrics_val_svm,
    "test_linear_svm": metrics_test_svm,
}
with open("test1/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

summary_rows = []
for split in ["val", "test"]:
    for model_name, m in [("logreg", metrics[f"{split}_logreg"]), ("linear_svm", metrics[f"{split}_linear_svm"])]:
        summary_rows.append({
            "split": split,
            "model": model_name,
            **m
        })
summary = pd.DataFrame(summary_rows)
summary.to_csv("test1/metrics_summary.csv", index=False)

print(summary)
