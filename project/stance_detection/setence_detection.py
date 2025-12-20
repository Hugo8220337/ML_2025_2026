import json
from pathlib import Path
from common.tools import export_model_results_csv, read_csv, load

from common.supervised_models import train_linear_regression, train_logistic_regression, train_svm
from common.nlp import tfidf_vectorize

# DATA_PATH = 'datasets/clickbait_data.csv'
DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "clickbait_data.csv"


def train_setence_detectuin_linear_regression(X, y):
    with load("Train Linear Regression Model..."):
        linear_regression_results = train_linear_regression(X, y)
    print("Linear Regression model training complete.")
    print(json.dumps(linear_regression_results['metrics']['Accuracy'], indent=4))
    
    return linear_regression_results

def train_setence_detection_logistic_regression(X, y):
    with load("Train Logistic Regression Model..."):
        logistic_regression_results = train_logistic_regression(X, y)
    print("Logistic Regression model training complete.")
    print(json.dumps(logistic_regression_results['metrics']['Accuracy'], indent=4))
    
    return logistic_regression_results

def train_setence_detection_svm(X, y):
    with load("Train SVM Model..."):
        svm_results = train_svm(X, y)
    print("SVM model training complete.")
    print(json.dumps(svm_results['metrics']['Accuracy'], indent=4))
    
    return svm_results

def setence_detection():
    df = read_csv(DATA_PATH)
    if df is None:
        raise FileNotFoundError(f"CSV não encontrado em {DATA_PATH}")
    
    dx = df["headline"]
    dy = df["clickbait"]

    X, vectorizer = tfidf_vectorize(df, col_name='headline')

    linear_regression_results = train_setence_detectuin_linear_regression(X, dy)
    logistic_regression_results = train_setence_detection_logistic_regression(X, dy)
    svm_results = train_setence_detection_svm(X, dy)

    all_results = {
        "Linear Regression": linear_regression_results,
        "Logistic Regression": logistic_regression_results,
        "SVM": svm_results
    }

    export_model_results_csv(all_results, "setence_detection_results.csv")