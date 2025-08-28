# Implementation & Comparative Analysis of Different Machine Learning Models

> **Hochschule Rhein‑Waal research project** benchmarking five supervised algorithms on an imbalanced healthcare stroke dataset. The repo contains the full data‑prep pipeline, training notebooks, trained models, and reproducible results.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](#requirements)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

## ✨ Key Findings

| Model                | Accuracy    |
| -------------------- | ----------- |
| Random Forest        | **99.44 %** |
| Decision Tree        | 99.22 %     |
| ANN (Keras)          | 98.54 %     |
| K‑Nearest Neighbours | 96.84 %     |
| Logistic Regression  | 79.53 %     |

> After SMOTE balancing and RFE/ANOVA feature selection, **Random Forest** achieved the best precision‑recall balance while training in < 3 s on a standard laptop.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Quick Start](#quick-start)
5. [Repo Structure](#repo-structure)
6. [Results & Visuals](#results--visuals)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

This repository underpins my paper **“Implementation & Comparative Analysis of Different Machine Learning Models.”** Five classifiers—Logistic Regression, KNN, Decision Tree, Random Forest and an Artificial Neural Network—are evaluated on a healthcare stroke dataset (original ≈ 5 110 rows, balanced ≈ 33 264). Steps include:

* Rigorous preprocessing & exploratory data analysis (EDA)
* Class balancing with SMOTE
* Feature selection via Recursive Feature Elimination (RFE) & ANOVA‑F
* Hyper‑parameter tuning with GridSearchCV (scikit‑learn)
* Comparative evaluation across five performance metrics

---

## Dataset

| File                                               | Description                        |
| -------------------------------------------------- | ---------------------------------- |
| `data/healthcare-dataset-stroke-data.csv`          | Raw stroke dataset from Kaggle     |
| `data/healthcare-dataset-stroke-data-balanced.csv` | Processed & SMOTE‑balanced version |

* **Source:** [Kaggle – “Stroke Prediction Dataset”](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* **Original shape:** 5 110 × 12 with a 95 : 5 class imbalance
* **Balanced shape:** 33 264 × 17 after SMOTE and targeted up‑sampling for *hypertension* & *heart\_disease*

> *Licence note:* the dataset is released under **CC0 1.0 Public Domain Dedication** (see [Dataset Licence](#dataset-licence)).

---

## Methodology

1. **EDA** – distributions, correlation heatmap, outlier checks.
2. **Data Cleaning** – dropped rare gender categories, imputed missing BMI with group means.
3. **Encoding** – binary and one‑hot for categorical attributes.
4. **Balancing** – SMOTE for the minority *stroke* class, stratified re‑sampling for comorbidities.
5. **Feature Selection** – RFE + ANOVA‑F → top 13 predictors.
6. **Model Training** – 80 / 20 split; scikit‑learn pipelines; ANN built in Keras/TensorFlow.
7. **Evaluation** – accuracy, precision, recall, F1, ROC‑AUC, training & inference times.

---

## Quick Start

```bash
# 1 Clone repo
git clone https://github.com/<your‑username>/implementation-and-comparative-analysis-of-different-machine-learning-models.git
cd implementation-and-comparative-analysis-of-different-machine-learning-models

# 2 Create environment & install deps
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3 Run notebook
jupyter notebook notebooks/Implementation_and_Analysis.ipynb
```

### Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
tensorflow   # Keras backend for ANN
```

---

## Repo Structure

```
│  README.md
│  requirements.txt
│
├─data/
│    ├─healthcare-dataset-stroke-data.csv
│    └─healthcare-dataset-stroke-data-balanced.csv
│
├─models/
│    ├─trained_ann_model.h5
│    └─random_forest.joblib
│
├─results/
│    ├─RF_sample_results.csv
│    ├─ANN_model_sample_results.csv
│    ├─DT_classifier_sample_results.csv
│    ├─KNN_classifier_sample_results.csv
│    └─LogReg_model_sample_results.csv
│
└─notebooks/
     └─Implementation_and_Analysis.ipynb
```

---

## Results & Visuals

The notebook includes full metric tables, confusion matrices, ROC curves and feature‑importance plots. **Takeaway:** ensemble methods (Random Forest) generalise best on balanced, multi‑feature healthcare data, while linear models trade accuracy for interpretability and speed.

---

## Contributing

Pull requests are welcome! Please open an issue to discuss major changes before submitting.

---
