# 🏦 Loan Default Prediction Pipeline

A production-ready machine learning pipeline for predicting loan defaults, built with scikit-learn.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Lessons Learned](#lessons-learned)

## 🚀 Overview

This project implements an end-to-end solution for predicting whether a loan applicant will default on their loan. The pipeline handles data preprocessing, custom feature engineering, model training, and evaluation - all packaged into a reusable scikit-learn pipeline.

## ✨ Key Features

- **Robust preprocessing** for numerical and categorical data
- **Custom feature engineering** including income-to-loan ratio calculation
- **Advanced class imbalance handling** using SMOTE
- **Hyperparameter optimization** for model performance
- **Production-ready design** with model serialization/deserialization
- **Comprehensive metrics** (accuracy, precision, recall, F1-score)

## 📊 Performance

The final model achieves strong predictive performance:

- High precision in identifying potential defaulters
- Balanced handling of the class imbalance problem
- Robust cross-validation results

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/qbakom/loan-default-pipeline.git
cd loan-default-pipeline

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage

**Train a new model:**
```bash
python3 train.py --target_column Status
```

**Make predictions with a trained model:**
```bash
python apply.py --pipeline_path models/pipeline_YYYYMMDD_HHMMSS.joblib \
                --data_path ../../data/Loan_Default.csv \
                --target_column Status \
                --output_path predictions.csv
```

## 🔬 Technical Implementation

- **Framework:** scikit-learn pipeline architecture
- **Preprocessing:** Automatic feature type detection, missing value imputation, scaling
- **Feature Engineering:** Custom transformers extending scikit-learn's BaseEstimator
- **Model:** Ensemble-based classification with RandomForest
- **Serialization:** joblib for model persistence

## 📝 Lessons Learned

- Ensuring column consistency between training and inference
- Preventing data leakage in preprocessing steps
- Importance of pipeline design for production deployments
- Managing rare classes with synthetic data generation


