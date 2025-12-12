import json
import pandas as pd
import numpy as np
from common.tools import read_csv
from common.supervised_models import train_linear_regression
from common.nlp import preprocessing, tfidf_vectorize


_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

def topic_classification():
    file_path = 'datasets/AGNEWS/train.csv'
    df = read_csv(file_path)
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])


    X, vectorizer = tfidf_vectorize(df, col_name='data')

    result = train_linear_regression(X, df['Class Index'])
    
    print(result)


    





