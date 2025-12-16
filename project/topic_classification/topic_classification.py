import json
import pandas as pd
from common.tools import read_csv, load
from common.supervised_models import train_linear_regression, train_logistic_regression, train_svm
from common.nlp import tfidf_vectorize


_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def topic_classification():
    df_train = read_csv('datasets/AGNEWS/train.csv')
    df_test = read_csv('datasets/AGNEWS/test.csv')
    df = pd.concat([df_train, df_test])
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])

    X, vectorizer = tfidf_vectorize(df, col_name='data')

    with load("Training Linear Regression model..."):
        linear_regression_results = train_linear_regression(X, df['Class Index'])
    print("Linear Regression model training complete.")
    print(json.dumps(linear_regression_results['metrics']['Accuracy'], indent=4))

    with load("Training Logistic Regression model..."):
        logistic_regression_results = train_logistic_regression(X, df['Class Index'])
    print("Logistic Regression model training complete.")
    print(json.dumps(logistic_regression_results['metrics']['Accuracy'], indent=4))

    with load("Training SVM model..."):
        svm_results = train_svm(X, df['Class Index'])
    print("SVM model training complete.")
    print(json.dumps(svm_results['metrics']['Accuracy'], indent=4))
    

    


    
