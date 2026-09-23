"""
train_models.py
Reproducible TF-IDF + multi-label classifier training pipeline.

Paper-aligned configuration:
- 80/20 held-out split
- TF-IDF unigrams + bigrams
- max_features=5000
- min_df=2
- sublinear_tf=True
- One-vs-Rest
- class_weight="balanced"
- Logistic Regression, Linear SVM, Random Forest
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from preprocess import prepare_dataset, LABELS


DATA_PATH = os.path.join("..", "data", "processed_dataset.csv")
RESULTS_DIR = os.path.join("..", "results")
MODEL_DIR = os.path.join("..", "results", "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def main():
    df, X_text, y = prepare_dataset(DATA_PATH)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Logistic Regression": OneVsRestClassifier(
            LogisticRegression(C=5.0, class_weight="balanced",
                               max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Linear SVM": OneVsRestClassifier(
            LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    }

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.joblib"))

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    pd.DataFrame({
        "report_text": X_test_text.reset_index(drop=True)
    }).to_csv(os.path.join(RESULTS_DIR, "test_text.csv"), index=False)

    y_test.reset_index(drop=True).to_csv(
        os.path.join(RESULTS_DIR, "test_labels.csv"), index=False
    )

    print("\nTraining complete.")
    print(f"Cleaned dataset size: {len(df)}")
    print(f"Training size: {len(X_train_text)}")
    print(f"Test size: {len(X_test_text)}")


if __name__ == "__main__":
    main()
