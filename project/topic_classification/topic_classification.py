import pandas as pd
import numpy as np
from common.tools import read_csv
from common.supervised_models import train_neural_network
from common.nlp import preprocessing


# Class Index,Title,Description


def topic_classification():
    file_path = 'datasets/AGNEWS/train.csv'
    df = read_csv(file_path)
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])


    df_test = read_csv(file_path)
    df_test['data'] = df_test['Title'] + ' ' + df_test['Description']
    df_test = df_test.drop(columns=['Title', 'Description'])
    df_test = df_test.iloc[[0]].copy()

    print(df_test['data'][0])

    df_test = preprocessing(df_test)

    print(df_test['data'][0])


    





