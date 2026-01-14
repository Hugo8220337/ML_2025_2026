import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common.nlp import tfidf_vectorize
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager
from common.visualizations import plot_model_comparison, plot_confusion_matrices

def clickbait_detection(
    models=['xgboost', 'cnn'],
    target_metric='f1_score',
    reduction=None,
    options='default',
    vectorizer_type='tfidf',
    visualizations=False
    ):

    cache = CacheManager(module_name='clickbait_detection')
    file_path = 'datasets/clickbait/'

    

    def preprocessing(file_path):
        df = read_csv(file_path+'clickbait_data.csv')
        return df



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)

    X = df['headline']
    y = df['clickbait']


    result = cache.execute(task_name='ems',
                           func=lambda: ems(
                            X=X,
                            y=y, 
                            models=models,
                            target_metric=target_metric,
                            report=True, 
                            options=options, 
                            reduction=reduction, 
                            vectorizer_type=vectorizer_type),
                           inputs=[X, y],
                           params={'models': models, 'options': options, 'reduction': reduction, 'vectorizer_type': vectorizer_type})

    with open('output.txt', 'w') as f:
        print(result, file=f)


    if visualizations:
        viz_dir = 'files/visualizations/clickbait_detection'
        os.makedirs(viz_dir, exist_ok=True)

        plot_model_comparison(result, save_path=os.path.join(viz_dir, 'model_comparison.png'))

        plot_confusion_matrices(result, save_path=os.path.join(viz_dir, 'confusion_matrices.png'))

    return result
        