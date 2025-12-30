import json
from pathlib import Path

import numpy as np
import pandas as pd
from common.nlp import preprocessing, tfidf_vectorize
from common.supervised_models import train_logistic_regression, train_svm
from common.tools import export_model_results_csv, load, read_csv
from sklearn.utils.class_weight import compute_class_weight

BODY_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "fnc1" / "fnc1_train_bodies.csv"
)
STANCES_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "fnc1" / "fnc1_train_stances.csv"
)


def _train_consistency_logistic(X, y, class_weights_dict):
    """Train Logistic Regression accepting class weights."""
    with load("Train Semantic Logistic Regression..."):
        # Passed 'class_weight' to the model to give importance to rare classes
        results = train_logistic_regression(
            X,
            y,
            class_weight=class_weights_dict,
            max_iter=1000,  # Increase iterations because the problem is difficult
        )
    print("Logistic Regression training complete.")

    # Print F1 or Accuracy for debugging
    if results and "metrics" in results:
        print(json.dumps(results["metrics"].get("f1_weighted", "N/A"), indent=4))
    return results


def _train_consistency_svm(X, y, class_weights_dict):
    """Train SVM accepting class weights."""
    with load("Train Semantic SVM..."):
        results = train_svm(X, y, class_weight=class_weights_dict)
    print("SVM training complete.")
    if results and "metrics" in results:
        print(json.dumps(results["metrics"].get("f1_weighted", "N/A"), indent=4))
    return results


def stance_detection():
    bodies = read_csv(BODY_DATA_PATH)
    stances = read_csv(STANCES_DATA_PATH)

    # Ensure the CSV reads succeeded before attempting to merge
    if bodies is None:
        raise RuntimeError(f"Failed to read bodies data from {BODY_DATA_PATH}")
    if stances is None:
        raise RuntimeError(f"Failed to read stances data from {STANCES_DATA_PATH}")

    df = pd.merge(stances, bodies, on="Body ID")  # Merge on Body ID

    # Map labels to numbers
    label_mapping = {"agree": 0, "disagree": 1, "discuss": 2, "unrelated": 3}
    # Use replace to avoid type-checking issues with map overloads
    df["label"] = df["Stance"].replace(label_mapping)

    # Verify if there are any unmapped labels (NaN values), remove them if found
    if bool(df["label"].isnull().any()):
        df = df.dropna(subset=["label"])

    # Fuse Title + Body
    df["combined_text"] = df["Headline"].fillna("") + " " + df["articleBody"].fillna("")

    # NLP Preprocessing: Tokenization, Lemmatization
    print("Applying NLP Preprocessing (Tokenization, Lemmatization)...")
    df["combined_text"] = preprocessing(df["combined_text"], remove_stopwords=False)
    print("Preprocessing complete.")

    # Vectorization, TF-IDF. It creates a matrix of features from the text data.
    print("Vectorizing data (TF-IDF)...")
    X, _vectorizer = tfidf_vectorize(
        df, col_name="combined_text", max_features=5000, lowercase=True
    )
    y = df["label"].astype(int)
    print("Vectorization complete.")

    # Compute class weights to handle class imbalance
    print("Computing class weights...")
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights_dict = dict(zip(classes, weights))
    print(f"Applied weights: {class_weights_dict}")

    # Train models
    logistic_results = _train_consistency_logistic(X, y, class_weights_dict)
    svm_results = _train_consistency_svm(X, y, class_weights_dict)

    # Export results
    all_results = {
        "Semantic Logistic Regression": logistic_results,
        "Semantic SVM": svm_results,
    }
    output_path = Path(__file__).resolve().parent / "semantic_consistency_results.csv"
    export_model_results_csv(all_results, str(output_path))
