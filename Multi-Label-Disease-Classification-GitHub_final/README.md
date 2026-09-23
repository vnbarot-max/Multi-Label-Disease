# Multi-Label Disease Classification from Radiology Reports

This repository contains the implementation associated with the research paper:

**Multi-Label Disease Classification from Radiology Reports Using Natural Language Processing: A Study on the Indiana University Chest X-ray Dataset**

## Overview

The project formulates radiology-report disease identification as a **multi-label text classification** problem.

The pipeline uses:

1. Radiology report text
2. Text cleaning and de-duplication
3. TF-IDF feature extraction
4. One-vs-Rest multi-label classification
5. Logistic Regression
6. Linear SVM
7. Random Forest
8. Multi-label evaluation using F1, precision, recall, Hamming loss and ROC-AUC

The study uses the findings/impression text associated with the Indiana University chest X-ray report corpus. The image pixels are not used in this text-classification pipeline.

## Dataset

Place the CSV file at:

```text
data/processed_dataset.csv
```

The supplied dataset contains the report text and disease-label columns used by the project.

The preprocessing script performs de-duplication before train/test splitting.

> Note: The dataset should be shared only if redistribution is permitted by the original dataset/license terms. Do not upload restricted patient-level or protected data.

## Disease Labels

The 15 labels used in the paper are:

- No Finding
- Calcified Granuloma
- Cardiomegaly
- Pulmonary Atelectasis
- Scoliosis Deformity
- Cicatrix Scarring
- Pleural Effusion
- Consolidation
- Nodule Mass
- Emphysema
- Fracture
- Edema
- Pneumonia
- Pneumothorax
- Fibrosis

## Project Structure

```text
Multi-Label-Disease-Classification/
│
├── data/
│   └── processed_dataset.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── predict.py
│
├── results/
│   ├── models/
│   ├── plots/
│   ├── summary.csv
│   └── per_label_results.csv
│
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

## Run the Pipeline

Open a terminal in the `src` folder.

### Step 1: Train models

```bash
python train_models.py
```

### Step 2: Evaluate models

```bash
python evaluate.py
```

The following files will be generated:

```text
results/summary.csv
results/per_label_results.csv
results/plots/model_comparison.png
results/models/
```

### Step 3: Predict a new report

```bash
python predict.py "The heart is enlarged with mild pulmonary vascular congestion."
```

## Reproducibility

The implementation uses a fixed random seed (`42`) for the main data split and model configuration.

The paper reports a cleaned corpus of 3063 reports and a held-out test set of 613 reports. When reproducing the paper, verify that your CSV contains the same source records and that the cleaning/de-duplication procedure produces the same sample counts.

## Reported Paper Results

The manuscript reports the following held-out test-set results:

| Model | Micro-F1 | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8431 | 0.7303 | 0.9842 |
| Linear SVM | 0.8468 | 0.7366 | 0.9878 |
| Random Forest | 0.6421 | 0.2905 | 0.9774 |

These numbers are the values reported in the manuscript. They should not be treated as newly generated results unless the code reproduces them on the same data split and preprocessing pipeline.

## Citation

If you use this repository, please cite the associated paper.

## Author

**Vishakha Barot**

Department of Computer Science & Engineering  
Karnavati University, India
