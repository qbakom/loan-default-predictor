#!/usr/bin/env python
"""
Zawiera:
1. Wczytywanie i eksplorację danych
2. Preprocessing numerycznych i kategorycznych cech
3. Inżynierię cech i customowe transformatory
4. Integrację modelu, trenowanie i ewaluację
5. Zapisywanie oraz ponowne wykorzystanie pipeline’u
6. Obsługę logowania, błędów oraz argumenty CLI
7. Testy jednostkowe i dokumentację
"""

import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer – dodaje nowe cechy.
    Przykład: jeśli w danych istnieją kolumny 'income' i 'loan_amount',
    tworzy cechę 'income_to_loan_ratio' jako stosunek dochodu do kwoty kredytu.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'income' in X.columns and 'loan_amount' in X.columns:
            X['income_to_loan_ratio'] = X['income'] / X['loan_amount'].replace(0, np.nan)
        return X

def load_and_explore_data(file_path):
    """
    Wczytuje dane z pliku CSV i wykonuje podstawową eksplorację:
    - typy danych, statystyki opisowe, brakujące wartości
    - zapisuje przykładowe wizualizacje rozkładu danych
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dane wczytane z {file_path}")
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania danych z {file_path}: {e}")
        sys.exit(1)
    
    logger.info("Typy danych:\n%s", df.dtypes)
    logger.info("Statystyki opisowe:\n%s", df.describe(include='all'))
    logger.info("Liczba brakujących wartości:\n%s", df.isnull().sum())

    try:
        plt.figure(figsize=(10, 6))
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            sns.histplot(df[num_cols[0]].dropna(), kde=True)
            plt.title(f"Rozkład: {num_cols[0]}")
            plt.savefig("num_distribution.png")
            plt.close()
        if len(df.select_dtypes(include=['object']).columns) > 0:
            plt.figure(figsize=(10, 6))
            cat_col = df.select_dtypes(include=['object']).columns[0]
            sns.countplot(x=cat_col, data=df)
            plt.title(f"Liczebność kategorii: {cat_col}")
            plt.savefig("cat_count.png")
            plt.close()
        logger.info("Wizualizacje zapisane jako 'num_distribution.png' oraz 'cat_count.png'")
    except Exception as e:
        logger.warning("Błąd podczas tworzenia wizualizacji: %s", e)

    return df

def build_preprocessing_pipeline(df):
    """
    Buduje pipeline przetwarzania danych – wykrywa automatycznie kolumny numeryczne i kategoryczne,
    a następnie tworzy odpowiednie pod-pipeline’y.
    """
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    logger.info("Kolumny numeryczne: %s", numerical_columns)
    logger.info("Kolumny kategoryczne: %s", categorical_columns)

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])

    return preprocessor, numerical_columns, categorical_columns

def train_and_evaluate_pipeline(df, preprocessor, custom_transformer, target_column, test_size=0.2, random_state=42):
    """
    Integruje preprocessing z modelem predykcyjnym (LogisticRegression), dzieli dane na zbiór treningowy i testowy,
    trenuje pipeline, dokonuje predykcji oraz ewaluacji (accuracy, precision, recall, confusion matrix).
    Wyniki zapisywane są w pliku evaluation_results.txt.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    full_pipeline = Pipeline([
        ('custom_features', custom_transformer),
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    full_pipeline.fit(X_train, y_train)
    logger.info("Trening pipeline’u zakończony")

    y_pred = full_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    logger.info("Accuracy: %s", acc)
    logger.info("Precision: %s", prec)
    logger.info("Recall: %s", rec)
    logger.info("Confusion Matrix:\n%s", conf_matrix)
    logger.info("Classification Report:\n%s", class_report)

    with open("evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\nClassification Report:\n")
        f.write(class_report)
    
    logger.info("Wyniki ewaluacji zapisane do evaluation_results.txt")
    return full_pipeline

def save_pipeline(pipeline, filename):
    """Zapisuje pipeline do pliku."""
    joblib.dump(pipeline, filename)
    logger.info("Pipeline zapisany do %s", filename)

def load_pipeline(filename):
    """Ładuje pipeline z pliku."""
    try:
        pipeline = joblib.load(filename)
        logger.info("Pipeline załadowany z %s", filename)
        return pipeline
    except Exception as e:
        logger.error("Błąd podczas ładowania pipeline’u z %s: %s", filename, e)
        return None

def run_unit_tests():
    """Uruchamia proste testy jednostkowe dla kluczowych funkcji."""
    import unittest

    class PipelineTestCase(unittest.TestCase):
        def setUp(self):
            self.df = pd.DataFrame({
                'income': [50000, 60000, 55000],
                'loan_amount': [200000, 250000, 230000],
                'credit_score': [700, 720, 710],
                'gender': ['M', 'F', 'F'],
                'default': [0, 1, 0]
            })

        def test_custom_feature_engineer(self):
            transformer = CustomFeatureEngineer()
            transformed = transformer.fit_transform(self.df)
            self.assertIn('income_to_loan_ratio', transformed.columns)

        def test_preprocessing_pipeline(self):
            preprocessor, _, _ = build_preprocessing_pipeline(self.df)
            transformed = preprocessor.fit_transform(self.df)
            self.assertIsNotNone(transformed)

    suite = unittest.TestLoader().loadTestsFromTestCase(PipelineTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

def main(args):
    df = load_and_explore_data(args.dataset_path)
    preprocessor, _, _ = build_preprocessing_pipeline(df)
    custom_transformer = CustomFeatureEngineer()
    pipeline_model = train_and_evaluate_pipeline(df, preprocessor, custom_transformer,
                                                 target_column=args.target_column,
                                                 test_size=args.test_size,
                                                 random_state=args.random_state)
    save_pipeline(pipeline_model, args.output_pipeline)

    loaded_pipeline = load_pipeline(args.output_pipeline)
    if loaded_pipeline:
        sample_predictions = loaded_pipeline.predict(df.drop(args.target_column, axis=1))
        logger.info("Przykładowe predykcje na pełnym zbiorze danych:\n%s", sample_predictions)

    if args.run_tests:
        run_unit_tests()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production-Ready Data Processing Pipeline")
    parser.add_argument('--dataset_path', type=str, default='../../data/Loan_Default.csv', help='Ścieżka do pliku CSV z danymi.')
    parser.add_argument('--target_column', type=str, default='default', help='Nazwa kolumny celu.')
    parser.add_argument('--output_pipeline', type=str, default='final_pipeline.pkl', help='Plik wyjściowy dla zapisanego pipeline’u.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proporcja zbioru testowego.')
    parser.add_argument('--random_state', type=int, default=42, help='Seed dla podziału zbioru.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Poziom logowania (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--run_tests', action='store_true', help='Uruchom testy jednostkowe po wykonaniu pipeline’u.')
    
    args = parser.parse_args()
    main(args)
