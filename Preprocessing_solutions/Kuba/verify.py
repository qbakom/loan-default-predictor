#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def feature_importance_plot(pipeline, feature_names, output_path="feature_importance.png"):
    """Create a feature importance plot for tree-based models."""
    if not hasattr(pipeline[-1], 'feature_importances_'):
        logger.warning("Model doesn't support feature importances.")
        return
    
    # Get feature importances
    importances = pipeline[-1].feature_importances_
    
    # If we have transformed feature names, try to use them
    if len(importances) != len(feature_names):
        logger.warning(f"Feature count mismatch: {len(importances)} importances vs {len(feature_names)} names")
        # Just use indices as feature names
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    plt.title("Feature Importances")
    plt.barh(range(min(20, len(indices))), importances[indices][:20], align="center")
    plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Feature importance plot saved to {output_path}")

def plot_roc_curve(y_true, y_proba, output_path="roc_curve.png"):
    """Plot ROC curve for model evaluation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    logger.info(f"ROC curve saved to {output_path}")

def threshold_optimization(y_true, y_proba, output_path="threshold_optimization.png"):
    """Find optimal threshold for classification."""
    thresholds = np.linspace(0.1, 0.9, 9)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
        precision_scores.append(precision_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    plt.plot(thresholds, precision_scores, 'g-', label='Precision')
    plt.plot(thresholds, recall_scores, 'r-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    
    # Find optimal threshold for F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f"Optimal threshold: {optimal_threshold:.2f} (F1: {f1_scores[optimal_idx]:.4f})")
    
    return optimal_threshold

def main(args):
    try:
        logger.info(f"Loading pipeline from {args.pipeline_path}")
        pipeline = joblib.load(args.pipeline_path)
        
        logger.info(f"Loading data from {args.data_path}")
        test_data = pd.read_csv(args.data_path)
        
        # Remove ID if present as it's not a predictive feature
        if 'ID' in test_data.columns:
            test_data = test_data.drop('ID', axis=1)
        
        if args.target_column in test_data.columns:
            X_test = test_data.drop(columns=[args.target_column])
            y_test = test_data[args.target_column]
            
            logger.info("Making predictions...")
            predictions = pipeline.predict(X_test)
            
            # Evaluate model
            logger.info("Evaluating model performance...")
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig('verification_confusion_matrix.png')
            logger.info("Confusion matrix saved as 'verification_confusion_matrix.png'")
            
            # If the model supports predict_proba, plot ROC curve and optimize threshold
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(X_test)[:, 1]
                
                # Plot ROC curve
                plot_roc_curve(y_test, probabilities)
                
                # Find optimal threshold
                optimal_threshold = threshold_optimization(y_test, probabilities)
                
                # Make predictions with optimal threshold
                optimized_predictions = (probabilities >= optimal_threshold).astype(int)
                optimized_f1 = f1_score(y_test, optimized_predictions)
                logger.info(f"F1 score with optimized threshold: {optimized_f1:.4f}")
                
                # Save results
                results = pd.DataFrame({
                    'actual': y_test,
                    'predicted': predictions,
                    'probability': probabilities,
                    'optimized_prediction': optimized_predictions
                })
            else:
                results = pd.DataFrame({
                    'actual': y_test,
                    'predicted': predictions
                })
                
            # Try to get feature names and plot importance
            try:
                feature_names = X_test.columns.tolist()
                feature_importance_plot(pipeline, feature_names)
            except Exception as e:
                logger.warning(f"Could not create feature importance plot: {str(e)}")
                
        else:
            # If no target column is available, just make predictions
            predictions = pipeline.predict(test_data)
            results = pd.DataFrame({'predicted': predictions})
            
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(test_data)
                for i in range(probabilities.shape[1]):
                    results[f'probability_class_{i}'] = probabilities[:, i]
        
        # Save results
        results.to_csv(args.output_path, index=False)
        logger.info(f"Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and analyze a saved ML pipeline")
    
    parser.add_argument("--pipeline_path", type=str, required=True,
                     help="Path to the saved pipeline file (.joblib)")
    parser.add_argument("--data_path", type=str, required=True,
                     help="Path to the verification data CSV file")
    parser.add_argument("--target_column", type=str, default="Status",
                     help="Name of the target column if present in the verification data")
    parser.add_argument("--output_path", type=str, default="verification_results.csv",
                     help="Path to save the verification results")
    
    args = parser.parse_args()
    main(args)
