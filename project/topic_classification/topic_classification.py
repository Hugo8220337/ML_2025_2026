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

    print(df['data'][0])
    
    df = preprocessing(df)

    print(df['data'][0])



    





