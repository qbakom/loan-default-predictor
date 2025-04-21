Machine Learning Pipeline Documentation
This project implements a comprehensive machine learning pipeline for loan default prediction, with modular components that work together. Below is an explanation of how each file works and how they communicate with each other.


## File structure

1. [`train_ext.py`](../train_ext.py) 

Purpose: Main training script that handles data preprocessing, feature engineering, model training and evaluation. Detailed comments included.

Key functionality:
- Loads and explores data with vis
- Identifies column types (numerical vs categorical)
- Applies feature engineering through CustomFeatureTransformer
- Trains models with options for 
    - Random forest and logistic regression
    - class imbalance handling (SMOTE)
    - Ensemble methods
    - AutoML optimization
    - Hyperparameter tuning
- Ealuates modele performance and saves the pipeline


1. `ensemble.models.py`

Add ensemble modeling capabilities for loan default prediction

- Create StackingEnsemble class that combines multiple models with meta-learning
- Implement voting ensembles with dynamic model integration
- Add two-stage model for classification followed by regression
- Handle optional dependencies (XGBoost, LightGBM, CatBoost)
- Fix regression model in two-stage approach

2. `train.py`

Add extended training pipeline with advanced features

- Enhance data exploration capabilities
- Add PCA and feature selection options
- Support ensemble models integration
- Add hyperparameter optimization
- Include advanced feature engineering techniques
- Connect with diagnostics module

3. `apply.py`

Add enhanced model application with extended capabilities

- Implement CustomFeatureTransformer for consistent preprocessing
- Add unique filename generation with timestamps
- Improve error handling and logging
- Support both classification and regression outputs
- Enhance results formatting with probabilities

4. `verify.py`
Implement model verification and performance analysis

- Add comprehensive metrics calculation
- Create confusion matrix visualizations
- Implement ROC curve analysis and plotting
- Add threshold optimization for classification models
- Generate feature importance visualizations
- Support models with and without probability outputs