"""
preprocess.py
Preprocessing utilities for multi-label radiology-report classification.
"""

import re
import pandas as pd


LABELS = [
    "No Finding",
    "Calcified Granuloma",
    "Cardiomegaly",
    "Pulmonary Atelectasis",
    "Scoliosis Deformity",
    "Cicatrix Scarring",
    "Pleural Effusion",
    "Consolidation",
    "Nodule Mass",
    "Emphysema",
    "Fracture",
    "Edema",
    "Pneumonia",
    "Pneumothorax",
    "Fibrosis",
]


def clean_text(text):
    """Lowercase text, remove de-identification placeholders and unwanted characters."""
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\bX{2,}\b", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataset(path):
    """Load, build report text if needed, clean, and remove exact duplicate reports."""
    df = pd.read_csv(path)

    if "report_text" not in df.columns:
        findings = df["findings"].fillna("") if "findings" in df.columns else ""
        impression = df["impression"].fillna("") if "impression" in df.columns else ""
        df["report_text"] = (findings.astype(str) + " " + impression.astype(str)).str.strip()

    df["report_text"] = df["report_text"].apply(clean_text)
    df = df[df["report_text"].str.len() > 0].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["report_text"]).reset_index(drop=True)
    print(f"Rows before de-duplication: {before}")
    print(f"Rows after de-duplication:  {len(df)}")

    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    X = df["report_text"]
    y = df[LABELS].fillna(0).astype(int)

    return df, X, y
