"""
evaluate.py
Evaluate trained multi-label classifiers and save paper-style result tables.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, precision_score, recall_score, hamming_loss,
    accuracy_score, roc_auc_score
)

from preprocess import LABELS

RESULTS_DIR = os.path.join("..", "results")
MODEL_DIR = os.path.join("..", "results", "models")


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def main():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    X_test_text = pd.read_csv(os.path.join(RESULTS_DIR, "test_text.csv"))["report_text"]
    y_test = pd.read_csv(os.path.join(RESULTS_DIR, "test_labels.csv"))

    X_test = vectorizer.transform(X_test_text)

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib",
        "Random Forest": "random_forest.joblib"
    }

    summary = []
    per_label_all = []

    for name, filename in model_files.items():
        model = joblib.load(os.path.join(MODEL_DIR, filename))
        y_pred = model.predict(X_test)
        scores = get_scores(model, X_test)

        try:
            macro_auc = roc_auc_score(y_test, scores, average="macro")
        except ValueError:
            macro_auc = np.nan

        summary.append({
            "Model": name,
            "Micro-F1": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Weighted-F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Micro-Precision": precision_score(y_test, y_pred, average="micro", zero_division=0),
            "Micro-Recall": recall_score(y_test, y_pred, average="micro", zero_division=0),
            "Hamming-Loss": hamming_loss(y_test, y_pred),
            "Exact-Match-Accuracy": accuracy_score(y_test, y_pred),
            "Macro-AUC": macro_auc
        })

        for i, label in enumerate(LABELS):
            try:
                auc = roc_auc_score(y_test.iloc[:, i], scores[:, i])
            except ValueError:
                auc = np.nan

            per_label_all.append({
                "Model": name,
                "Label": label,
                "Precision": precision_score(y_test.iloc[:, i], y_pred[:, i], zero_division=0),
                "Recall": recall_score(y_test.iloc[:, i], y_pred[:, i], zero_division=0),
                "F1": f1_score(y_test.iloc[:, i], y_pred[:, i], zero_division=0),
                "Support": int(y_test.iloc[:, i].sum()),
                "AUC": auc
            })

    summary_df = pd.DataFrame(summary)
    per_label_df = pd.DataFrame(per_label_all)

    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    per_label_df.to_csv(os.path.join(RESULTS_DIR, "per_label_results.csv"), index=False)

    # Paper-style comparison plot
    plot_df = summary_df.set_index("Model")[["Micro-F1", "Macro-F1", "Macro-AUC"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Score")
    ax.set_title("Multi-Label Classification Performance")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "model_comparison.png"), dpi=300)
    plt.close()

    print("\nOverall results:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
