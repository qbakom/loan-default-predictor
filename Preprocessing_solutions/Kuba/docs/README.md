# Enhanced ML Pipeline for Loan Default Prediction

This package provides a comprehensive machine learning pipeline for loan default prediction, inspired by top Kaggle competition approaches.

## Features

- **Advanced EDA**: Identify outliers, detect data leakage, and visualize feature distributions
- **Feature Engineering**: Create domain-specific features and apply transformations
- **Model Selection**: Choose from traditional models, ensembles, or AutoML approaches
- **Advanced Diagnostics**: Visualize model performance with Yellowbrick and explain predictions with SHAP
- **Ensembling**: Combine multiple models using voting, stacking, or two-stage approaches
- **AutoML Integration**: Automatically optimize pipelines with TPOT or auto-sklearn

## Installation

First, install the required dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn
```

For advanced features, install optional dependencies:

```bash
pip install yellowbrick shap xgboost lightgbm catboost tpot auto-sklearn
```

## Usage

### Basic Usage

```bash
python gc1_copy.py --data_path ../../data/Loan_Default.csv --target_column Status
```

### With Feature Engineering

```bash
python gc1_copy.py --data_path ../../data/Loan_Default.csv --target_column Status --cap_outliers --create_interactions
```

### With Ensemble Models

```bash
python gc1_copy.py --data_path ../../data/Loan_Default.csv --target_column Status --use_ensemble --ensemble_type stacking
```

### With Advanced Diagnostics

```bash
python gc1_copy.py --data_path ../../data/Loan_Default.csv --target_column Status --advanced_diagnostics
```

### With AutoML

```bash
python gc1_copy.py --data_path ../../data/Loan_Default.csv --target_column Status --use_automl --automl_backend tpot --automl_time_budget 60
```

### Using a Saved Pipeline

```bash
python gc2_copy.py --pipeline_path ./models/pipeline_20230101_120000.joblib --data_path ../../data/Loan_Default_test.csv --target_column Status
```

### Verifying Pipeline Performance

```bash
python gc3.py --pipeline_path ./models/pipeline_20230101_120000.joblib --data_path ../../data/Loan_Default_test.csv --target_column Status
```

## Main Components

1. **gc1_copy.py**: Main pipeline for data preprocessing, feature engineering, and model training
2. **gc2_copy.py**: Script to apply a saved pipeline to new data
3. **gc3.py**: Script to verify and analyze a saved pipeline
4. **ensemble_models.py**: Implementation of ensemble and stacking models
5. **advanced_diagnostics.py**: Tools for model diagnostics, visualization, and AutoML

## Command Line Arguments

### gc1_copy.py

- `--data_path`: Path to the input CSV file
- `--target_column`: Name of the target column
- `--model_type`: Type of model to use (random_forest or logistic)
- `--output_dir`: Directory to save model artifacts
- `--cap_outliers`: Whether to cap outliers in numerical features
- `--create_interactions`: Whether to create interaction terms between features
- `--handle_imbalance`: Whether to handle class imbalance using SMOTE
- `--tune_hyperparameters`: Whether to perform hyperparameter tuning
- `--use_ensemble`: Whether to use ensemble methods
- `--ensemble_type`: Type of ensemble to use (voting, stacking, two_stage)
- `--advanced_diagnostics`: Whether to run advanced diagnostics
- `--use_automl`: Whether to use AutoML optimization
- `--automl_backend`: AutoML backend to use (tpot or autosklearn)
- `--automl_time_budget`: Time budget for AutoML optimization in minutes
- `--apply_pca`: Whether to apply PCA for dimensionality reduction

### gc2_copy.py

- `--pipeline_path`: Path to the saved pipeline file (.joblib)
- `--data_path`: Path to the new data CSV file
- `--target_column`: Name of the target column if present in the new data
- `--output_path`: Path to save the predictions

### gc3.py

- `--pipeline_path`: Path to the saved pipeline file (.joblib)
- `--data_path`: Path to the verification data CSV file
- `--target_column`: Name of the target column if present in the verification data
- `--output_path`: Path to save the verification results
