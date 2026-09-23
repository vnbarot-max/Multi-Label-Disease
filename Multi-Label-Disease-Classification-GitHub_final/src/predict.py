"""
predict.py
Predict disease labels for a new radiology report.
"""

import os
import sys
import joblib
import numpy as np

from preprocess import LABELS

MODEL_DIR = os.path.join("..", "results", "models")


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py "radiology report text"')
        return

    text = " ".join(sys.argv[1:])

    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(MODEL_DIR, "linear_svm.joblib"))

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    print("\nPredicted findings:")
    found = False
    for label, value in zip(LABELS, prediction):
        if value == 1:
            print(f"- {label}")
            found = True

    if not found:
        print("- No positive label predicted")


if __name__ == "__main__":
    main()
