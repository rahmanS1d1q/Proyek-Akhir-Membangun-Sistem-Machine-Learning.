import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import dagshub

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# =========================================================
# 0) CONFIG DAGSHUB (SUDAH DIBENERIN)
# =========================================================
DAGSHUB_USERNAME = "rahmanshiddiq09"
DAGSHUB_REPO = "telco-churn-mlflow-advance"

# Inisialisasi DagsHub + aktifkan MLflow tracking online
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)

# =========================================================
# 1) LOAD DATASET
# =========================================================
X_train = pd.read_csv("namadataset_preprocessing/X_train.csv")
X_test = pd.read_csv("namadataset_preprocessing/X_test.csv")
y_train = pd.read_csv("namadataset_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("namadataset_preprocessing/y_test.csv").values.ravel()

# =========================================================
# 2) MLFLOW EXPERIMENT (ONLINE - DAGSHUB) - FIX 404
# =========================================================
EXPERIMENT_NAME = "Telco Churn Classification - Advance (DagsHub)"

client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)

if exp is None:
    exp_id = client.create_experiment(EXPERIMENT_NAME)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(experiment_id=exp_id)

# =========================================================
# 3) HYPERPARAMETER TUNING
# =========================================================
base_model = LogisticRegression(max_iter=2000)

param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["liblinear", "lbfgs"],
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

# =========================================================
# 4) RUN + MANUAL LOGGING (ADVANCE)
# =========================================================
run = mlflow.start_run(run_name="logreg_gridsearch_manual_logging_advance")
try:
    # Fit grid search
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    # Predict
    y_pred = best_model.predict(X_test)

    # Predict proba (untuk AUC, log_loss, ROC/PR curve)
    y_proba = None
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]

    # -------------------------
    # Manual logging: params
    # -------------------------
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_f1_best", float(best_cv_score))

    # -------------------------
    # Manual logging: metrics (autolog + ekstra)
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("test_accuracy", float(acc))
    mlflow.log_metric("test_precision", float(prec))
    mlflow.log_metric("test_recall", float(rec))
    mlflow.log_metric("test_f1", float(f1))

    # Metrics ekstra (beyond autolog)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    mlflow.log_metric("test_balanced_accuracy", float(bal_acc))
    mlflow.log_metric("test_mcc", float(mcc))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        ll = log_loss(y_test, y_proba)
        mlflow.log_metric("test_roc_auc", float(auc))
        mlflow.log_metric("test_log_loss", float(ll))

    # =========================================================
    # 5) ARTIFACTS (MINIMAL 2 TAMBAHAN)
    # =========================================================
    os.makedirs("artifacts", exist_ok=True)

    # A) Confusion matrix (artifact)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix (Test)")
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(cm_path)

    # B) Classification report (artifact)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_path = "artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path)

    # C) ROC curve (artifact tambahan #1)
    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve (Test)")
        roc_path = "artifacts/roc_curve.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(roc_path)

        # D) PR curve (artifact tambahan #2)
        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title("Precision-Recall Curve (Test)")
        pr_path = "artifacts/pr_curve.png"
        plt.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(pr_path)

    # =========================================================
    # 6) LOG MODEL (signature + input_example)
    # =========================================================
    input_example = X_train.head(5)
    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
    )

    mlflow.end_run(status="FINISHED")

    print("✅ Advance (DagsHub) run selesai")
    print("Best params:", best_params)
    print("Test accuracy:", acc)
    print("Test f1:", f1)

except Exception:
    mlflow.end_run(status="FAILED")
    raise
