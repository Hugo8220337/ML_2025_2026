import json
from pathlib import Path
from common.tools import read_csv, load

from common.supervised_models import train_linear_regression
from common.nlp import tfidf_vectorize

# DATA_PATH = 'datasets/clickbait_data.csv'
DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "clickbait_data.csv"


def setence_detection():
    df = read_csv(DATA_PATH)
    if df is None:
        raise FileNotFoundError(f"CSV não encontrado em {DATA_PATH}")
    
    dx = df["headline"]
    dy = df["clickbait"]

    X, vectorizer = tfidf_vectorize(df, col_name='headline')

    with load("Load Setence Detection Model..."):
        linear_regression_results = train_linear_regression(X, dy)
    print("Linear Regression model training complete.")
    print(json.dumps(linear_regression_results['metrics']['Accuracy'], indent=4))


if __name__ == "__main__":
    setence_detection()