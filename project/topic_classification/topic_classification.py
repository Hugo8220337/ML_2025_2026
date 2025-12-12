import json
import pandas as pd
import numpy as np
from common.tools import read_csv
from common.supervised_models import train_linear_regression, train_logistic_regression, train_svm
from common.nlp import preprocessing, tfidf_vectorize


_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

def topic_classification():
    df_train = read_csv('datasets/AGNEWS/train.csv')
    df_test = read_csv('datasets/AGNEWS/test.csv')
    df = pd.concat([df_train, df_test])
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])


    X, vectorizer = tfidf_vectorize(df, col_name='data')

    result1 = train_linear_regression(X, df['Class Index'])
    
    print(json.dumps(result1['metrics']['Accuracy'], indent=4))

    result2 = train_logistic_regression(X, df['Class Index'])
    
    print(json.dumps(result2['metrics']['Accuracy'], indent=4))

    result3 = train_svm(X, df['Class Index'])
    
    print(json.dumps(result3['metrics']['Accuracy'], indent=4))
    

    


    





