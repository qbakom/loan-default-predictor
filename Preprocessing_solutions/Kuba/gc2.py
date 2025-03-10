#!/usr/bin/env python3
"""
Example script to demonstrate how to use a saved preprocessing pipeline
"""

import joblib
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    """Load a saved pipeline and apply it to new data"""
    try:
        # Load the pipeline
        logger.info(f"Loading pipeline from {args.pipeline_path}")
        pipeline = joblib.load(args.pipeline_path)
        
        # Load new data
        logger.info(f"Loading data from {args.data_path}")
        new_data = pd.read_csv(args.data_path)
        
        # Apply the pipeline to new data
        logger.info("Applying pipeline to new data")
        if args.target_column in new_data.columns:
            # If target column exists, separate it
            X_new = new_data.drop(columns=[args.target_column])
            y_new = new_data[args.target_column]
            
            # Make predictions
            predictions = pipeline.predict(X_new)
            logger.info(f"Made {len(predictions)} predictions")
            
            # If this is a classification task, add probabilities
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(X_new)
                # Create a results DataFrame with predictions and probabilities
                results = pd.DataFrame({
                    'actual': y_new,
                    'predicted': predictions
                })
                
                # Add probability columns for each class
                for i in range(probabilities.shape[1]):
                    results[f'prob_class_{i}'] = probabilities[:, i]
            else:
                # For regression tasks
                results = pd.DataFrame({
                    'actual': y_new,
                    'predicted': predictions
                })
        else:
            # No target column, just make predictions
            predictions = pipeline.predict(new_data)
            results = pd.DataFrame({'predicted': predictions})
        
        # Save results
        results.to_csv(args.output_path, index=False)
        logger.info(f"Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a saved pipeline to new data")
    
    parser.add_argument("--pipeline_path", type=str, required=True,
                     help="Path to the saved pipeline file (.joblib)")
    parser.add_argument("--data_path", type=str, required=True,
                     help="Path to the new data CSV file")
    parser.add_argument("--target_column", type=str, default=None,
                     help="Name of the target column if present in the new data")
    parser.add_argument("--output_path", type=str, default="pipeline_results.csv",
                     help="Path to save the predictions")
    
    args = parser.parse_args()
    main(args)