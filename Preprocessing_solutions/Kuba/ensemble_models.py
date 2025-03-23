#!/usr/bin/env python3
"""
Ensemble and Stacking models for loan default prediction
Inspired by top Kaggle competition approaches
"""

import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

# Try to import specialized models
try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, ModuleNotFoundError):
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, ModuleNotFoundError):
    HAS_LGB = False
    logger.warning("LightGBM not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except (ImportError, ModuleNotFoundError):
    HAS_CATBOOST = False
    logger.warning("CatBoost not available")

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    A stacking ensemble that combines multiple base models and a meta-model.
    
    First, base models make predictions, then these predictions become
    features for the meta-model to make the final prediction.
    """
    
    def __init__(self, base_models=None, meta_model=None, use_probas=True, 
                 cv=5, use_features=False):
        """
        Initialize the stacking ensemble.
        
        Args:
            base_models (list): List of base models
            meta_model: Meta-model to make final prediction
            use_probas (bool): Whether to use probability predictions from base models
            cv (int): Cross-validation folds for creating meta-features
            use_features (bool): Whether to include original features in meta-model
        """
        self.base_models = base_models or [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            LogisticRegression(random_state=42)
        ]
        self.meta_model = meta_model or LogisticRegression(random_state=42)
        self.use_probas = use_probas
        self.cv = cv
        self.use_features = use_features
        
    def fit(self, X, y):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Target labels
            
        Returns:
            self: Returns self
        """
        # Train base models
        self.fitted_base_models_ = []
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            fitted_model = clone(model).fit(X, y)
            self.fitted_base_models_.append(fitted_model)
        
        # Create meta-features using cross-validation
        logger.info("Creating meta-features through cross-validation")
        meta_features = self._create_meta_features(X, y)
        
        # Include original features if specified
        if self.use_features:
            meta_features = np.hstack((meta_features, X))
        
        # Train meta-model
        logger.info("Training meta-model")
        self.meta_model_ = clone(self.meta_model).fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """
        Predict using the stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Predicted labels
        """
        meta_features = self._create_meta_features_test(X)
        
        # Include original features if specified
        if self.use_features:
            meta_features = np.hstack((meta_features, X))
        
        return self.meta_model_.predict(meta_features)
    
    def predict_proba(self, X):
        """
        Predict probabilities using the stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        meta_features = self._create_meta_features_test(X)
        
        # Include original features if specified
        if self.use_features:
            meta_features = np.hstack((meta_features, X))
        
        return self.meta_model_.predict_proba(meta_features)
    
    def _create_meta_features(self, X, y):
        """
        Create meta-features for training data using cross-validation.
        
        Args:
            X: Training features
            y: Target labels
            
        Returns:
            np.ndarray: Meta-features
        """
        meta_features = np.zeros((X.shape[0], len(self.base_models) * (2 if self.use_probas else 1)))
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Creating meta-features for model {i+1}/{len(self.base_models)}")
            if self.use_probas:
                # Check if model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    cv_probas = cross_val_predict(model, X, y, cv=self.cv, method='predict_proba')
                    meta_features[:, i*2:(i+1)*2] = cv_probas
                else:
                    # Use predict and expand to two columns for consistency
                    cv_preds = cross_val_predict(model, X, y, cv=self.cv)
                    meta_features[:, i*2] = cv_preds
                    meta_features[:, i*2+1] = 1 - cv_preds
            else:
                cv_preds = cross_val_predict(model, X, y, cv=self.cv)
                meta_features[:, i] = cv_preds
        
        return meta_features
    
    def _create_meta_features_test(self, X):
        """
        Create meta-features for test data using fitted base models.
        
        Args:
            X: Test features
            
        Returns:
            np.ndarray: Meta-features
        """
        meta_features = np.zeros((X.shape[0], len(self.base_models) * (2 if self.use_probas else 1)))
        
        for i, model in enumerate(self.fitted_base_models_):
            if self.use_probas:
                # Check if model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    test_probas = model.predict_proba(X)
                    meta_features[:, i*2:(i+1)*2] = test_probas
                else:
                    # Use predict and expand to two columns for consistency
                    test_preds = model.predict(X)
                    meta_features[:, i*2] = test_preds
                    meta_features[:, i*2+1] = 1 - test_preds
            else:
                test_preds = model.predict(X)
                meta_features[:, i] = test_preds
        
        return meta_features

def create_voting_ensemble(preprocessor):
    """
    Create a voting ensemble that combines multiple models.
    
    Args:
        preprocessor: Preprocessing pipeline
        
    Returns:
        Pipeline: Voting ensemble pipeline
    """
    # Create base classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    estimators = [('rf', rf), ('gb', gb), ('lr', lr)]
    
    # Add specialized models if available
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        estimators.append(('xgb', xgb_model))
    
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        estimators.append(('lgb', lgb_model))
    
    if HAS_CATBOOST:
        cb_model = cb.CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
        estimators.append(('cb', cb_model))
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])
    
    return pipeline

def create_stacking_ensemble(preprocessor, use_features=False):
    """
    Create a stacking ensemble pipeline.
    
    Args:
        preprocessor: Preprocessing pipeline
        use_features (bool): Whether to include original features in meta-model
        
    Returns:
        Pipeline: Stacking ensemble pipeline
    """
    # Create base models
    base_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        LogisticRegression(random_state=42, max_iter=1000)
    ]
    
    # Add specialized models if available
    if HAS_XGB:
        base_models.append(xgb.XGBClassifier(n_estimators=100, random_state=42))
    
    if HAS_LGB:
        base_models.append(lgb.LGBMClassifier(n_estimators=100, random_state=42))
    
    # Create meta-model
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create stacking ensemble
    stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        use_probas=True,
        cv=5,
        use_features=use_features
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stacking)
    ])
    
    return pipeline

def create_two_stage_model(preprocessor):
    """
    Create a two-stage model: first classify default/non-default,
    then estimate loss amount for defaults.
    
    Args:
        preprocessor: Preprocessing pipeline
        
    Returns:
        tuple: (classification_pipeline, regression_pipeline)
    """
    # Classification stage
    clf_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('classifier', clf_model)
    ])
    
    # Regression stage - FIXED: Now using a regressor instead of classifier
    reg_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    reg_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('regressor', reg_model)
    ])
    
    return clf_pipeline, reg_pipeline
