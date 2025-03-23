#!/usr/bin/env python3
"""
Advanced diagnostic tools for machine learning models
Includes Yellowbrick visualizations and AutoML integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import yellowbrick
    from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC
    from yellowbrick.model_selection import LearningCurve, ValidationCurve
    from yellowbrick.features import PCA as PCAViz, Rank
    HAS_YELLOWBRICK = True
except ImportError:
    logger.warning("Yellowbrick not available. Visualizations will be limited.")
    HAS_YELLOWBRICK = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    logger.warning("SHAP not available. Feature importance will use built-in methods.")
    HAS_SHAP = False

try:
    from tpot import TPOTClassifier
    HAS_TPOT = True
except ImportError:
    logger.warning("TPOT not available. AutoML functionality will be disabled.")
    HAS_TPOT = False

try:
    import autosklearn.classification
    HAS_AUTOSKLEARN = True
except ImportError:
    logger.warning("auto-sklearn not available. AutoML functionality will be limited.")
    HAS_AUTOSKLEARN = False

class ModelDiagnostics:
    """
    Class for visualizing and diagnosing machine learning models.
    """
    
    def __init__(self, output_dir="./diagnostics"):
        """
        Initialize the diagnostics class.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def yellowbrick_visualizations(self, model, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Create Yellowbrick visualizations for model diagnostics.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            feature_names (list): Names of features
        """
        if not HAS_YELLOWBRICK:
            logger.warning("Yellowbrick not available. Skipping visualizations.")
            return
        
        # Classification Report
        logger.info("Creating classification report visualization")
        viz = ClassificationReport(model, classes=['0', '1'], support=True)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.save(os.path.join(self.output_dir, "classification_report.png"))
        
        # Confusion Matrix
        logger.info("Creating confusion matrix visualization")
        viz = ConfusionMatrix(model, classes=['0', '1'])
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.save(os.path.join(self.output_dir, "confusion_matrix.png"))
        
        # ROC-AUC Curve
        if hasattr(model, "predict_proba"):
            logger.info("Creating ROC-AUC curve visualization")
            viz = ROCAUC(model, classes=['0', '1'])
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.save(os.path.join(self.output_dir, "roc_auc_curve.png"))
        
        # Learning Curve
        logger.info("Creating learning curve visualization")
        viz = LearningCurve(model, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 5))
        viz.fit(X_train, y_train)
        viz.save(os.path.join(self.output_dir, "learning_curve.png"))
        
        # Feature Importance (if available)
        if feature_names and hasattr(model, "feature_importances_"):
            logger.info("Creating feature importance visualization")
            viz = Rank(model, features=feature_names, show=False)
            viz.fit(X_train, y_train)
            viz.save(os.path.join(self.output_dir, "feature_importance.png"))
        
        # PCA Visualization
        if X_train.shape[1] > 2:  # Only if we have more than 2 features
            logger.info("Creating PCA visualization")
            viz = PCAViz(scale=True, projection=2)
            viz.fit_transform(X_train, y_train)
            viz.save(os.path.join(self.output_dir, "pca_visualization.png"))
    
    def shap_analysis(self, model, X_train, X_test, feature_names=None):
        """
        Create SHAP analysis for model interpretability.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            feature_names (list): Names of features
        """
        if not HAS_SHAP:
            logger.warning("SHAP not available. Skipping analysis.")
            return
        
        try:
            # Create a small sample for SHAP analysis to avoid memory issues
            X_sample = X_test[:100] if X_test.shape[0] > 100 else X_test
            
            # Try to get the underlying model if it's a pipeline
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
            else:
                classifier = model
            
            # Create explainer
            if hasattr(classifier, "predict_proba"):
                explainer = shap.TreeExplainer(classifier) if hasattr(classifier, "estimators_") else shap.LinearExplainer(classifier, X_train)
                shap_values = explainer.shap_values(X_sample)
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_summary.png"))
                plt.close()
                
                # Feature importance plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "shap_importance.png"))
                plt.close()
                
                logger.info("SHAP analysis completed and saved")
            else:
                logger.warning("Model doesn't support SHAP analysis")
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
    
    def auto_ml_optimization(self, X_train, y_train, X_test, y_test, time_budget=60, use_tpot=True):
        """
        Use AutoML to find the best model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            time_budget (int): Time budget in minutes
            use_tpot (bool): Whether to use TPOT or auto-sklearn
            
        Returns:
            object: Trained AutoML model
        """
        if use_tpot and HAS_TPOT:
            logger.info(f"Starting TPOT optimization with {time_budget} minute time budget")
            automl = TPOTClassifier(
                generations=5,
                population_size=20,
                verbosity=2,
                random_state=42,
                max_time_mins=time_budget,
                config_dict='TPOT sparse'
            )
            automl.fit(X_train, y_train)
            
            # Evaluate
            score = automl.score(X_test, y_test)
            logger.info(f"TPOT optimization completed. Test score: {score:.4f}")
            
            # Export pipeline
            automl.export(os.path.join(self.output_dir, "tpot_pipeline.py"))
            logger.info(f"TPOT pipeline exported to {os.path.join(self.output_dir, 'tpot_pipeline.py')}")
            
            return automl
        elif HAS_AUTOSKLEARN:
            logger.info(f"Starting auto-sklearn optimization with {time_budget} minute time budget")
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=time_budget * 60,  # convert to seconds
                per_run_time_limit=time_budget * 30,  # 30% of total time per run
                tmp_folder=os.path.join(self.output_dir, "autosklearn_tmp"),
                output_folder=os.path.join(self.output_dir, "autosklearn_out")
            )
            automl.fit(X_train, y_train)
            
            # Evaluate
            score = automl.score(X_test, y_test)
            logger.info(f"auto-sklearn optimization completed. Test score: {score:.4f}")
            
            # Show models
            logger.info("auto-sklearn leaderboard:")
            for i, (model, perf) in enumerate(automl.leaderboard().iterrows()):
                if i < 5:  # Show top 5 models
                    logger.info(f"{i+1}. {perf['model_id']}: {perf['cost']:.4f}")
            
            return automl
        else:
            logger.warning("No AutoML libraries available. Skipping AutoML optimization.")
            return None
