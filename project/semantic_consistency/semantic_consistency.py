import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from common.tools import read_csv

from common.tools import export_model_results_csv, read_csv, load
from common.nlp import tfidf_vectorize
from common.supervised_models import train_logistic_regression, train_svm

BODY_DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "fnc1" / "fnc1_train_bodies.csv"
STANCES_DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "fnc1" / "fnc1_train_stances.csv"

def _train_consistency_logistic(X, y, class_weights_dict):
    """Train Logistic Regression accepting class weights."""
    with load("Train Semantic Logistic Regression..."):
        # Passed 'class_weight' to the model to give importance to rare classes
        results = train_logistic_regression(
            X, y, 
            class_weight=class_weights_dict,
            max_iter=1000 # Increase iterations because the problem is difficult
        )
    print("Logistic Regression training complete.")

    # Print F1 or Accuracy for debugging
    if results and 'metrics' in results:
        print(json.dumps(results['metrics'].get('F1 Score', 'N/A'), indent=4))
    return results

def _train_consistency_svm(X, y, class_weights_dict):
    """Train SVM accepting class weights."""
    with load("Train Semantic SVM..."):
        results = train_svm(
            X, y, 
            class_weight=class_weights_dict
        )
    print("SVM training complete.")
    if results and 'metrics' in results:
        print(json.dumps(results['metrics'].get('F1 Score', 'N/A'), indent=4))
    return results

def semantic_consistency():
    bodies = read_csv(BODY_DATA_PATH)
    stances = read_csv(STANCES_DATA_PATH)

    # Merge bodies and stances on 'Body ID'
    df = pd.merge(stances, bodies, on='Body ID')

    # Map labels to numbers
    label_mapping = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

    # Create a new column 'label' with mapped values
    df['label'] = df['Stance'].map(label_mapping)

    # Verify if there are any unmapped labels (NaN values)
    if df['label'].isnull().any():
        df = df.dropna(subset=['label']) # Remove rows with unmapped labels

    # Feature Engineering (Fuse Title + Body)
    # Important: The TF-IDF needs to see the words from both to find relationships
    df['combined_text'] = df['Headline'] + " " + df['articleBody']

    # Vectorization (may take a while due to the size of FNC-1)
    print("Vectorizing data (TF-IDF)...")
    # max_features limit the size to avoid running out of RAM
    # i do not use stop owrds removal because TfidfVectorizer stop words for english includes the 'not' word, which is important for stance detection
    X, vectorizer = tfidf_vectorize(df, col_name='combined_text', max_features=5000, lowercase=True)
    y = df['label']

    # Compute class weights for balancing
    # Cannot use simple label encoding because there are few 'disagree' samples and the model might ignore them
    # To handle class imbalance, we compute balanced class weights
    print("Computing class weights...")
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
    # Convert array to dictionary that sklearn understands: {0: weight, 1: weight...}
    class_weights_dict = dict(zip(classes, weights))
    print(f"Applied weights: {class_weights_dict}")

    # Train Models
    logistic_results = _train_consistency_logistic(X, y, class_weights_dict)
    svm_results = _train_consistency_svm(X, y, class_weights_dict)

    # Export results
    all_results = {
        "Semantic Logistic Regression": logistic_results,
        "Semantic SVM": svm_results
    }

    # Define output path
    output_path = Path(__file__).resolve().parent / "semantic_consistency_results.csv"
    
    export_model_results_csv(all_results, str(output_path))


