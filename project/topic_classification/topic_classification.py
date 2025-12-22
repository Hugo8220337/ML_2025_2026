import json
import pandas as pd
from common.tools import read_csv, load
from common.supervised_models import train_linear_regression, train_logistic_regression, train_svm
from common.nlp import tfidf_vectorize
from evolutionary_model_selection.ems import ems



_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def topic_classification():
    df = read_csv('datasets/AGNEWS/train.csv')
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])

    X, vectorizer = tfidf_vectorize(df, col_name='data')

    result = ems(X, df['Class Index'], ['logistic_regression'], report=True)
    print(json.dumps(result['info'], indent=4))


    


    
