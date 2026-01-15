import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from common.nlp import tfidf_vectorize
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager
from common.visualizations import plot_model_comparison, plot_confusion_matrices
from fake_news_detection.pipeline import run_pipeline


def fake_news_detection(
    models=['logistic_regression', 'neural_network', 'svm', 'random_forest', 'xgboost'],
    target_metric='f1_score',
    reduction='nmf',
    options='default',
    vectorizer_type=None,
    visualizations=True):
    
    cache = CacheManager(module_name='fake_news_detection')
    file_path = 'datasets/WELFake/'

    labels = {
        'fake': 0,
        'real': 1
    }

    def preprocessing(file_path):
        df = read_csv(file_path+'WELFake_Dataset.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        return df



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)


    df = cache.execute(task_name='run_pipeline',
                       func=lambda: run_pipeline(df),
                       inputs=df)


    X_df = pd.DataFrame()
    X_df['data'] = df['title'].fillna('') + " " + df['text'].fillna('')
    X_df['topic'] = df['topic']
    X_df['stance'] = df['stance'].astype(str)
    X_df['anomaly'] = df['anomaly']
    X_df['clickbait'] = df['clickbait']

    y = df['label']

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), 'data'),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=2), ['topic', 'stance']),
            ('passthrough', 'passthrough', ['anomaly', 'clickbait'])
        ],
        remainder='drop' 
    )

    X = preprocessor.fit_transform(X_df)

    result = cache.execute(task_name='ems',
                           func=lambda: ems(
                            X=X,
                            y=y, 
                            models=models,
                            target_metric=target_metric,
                            report=True, 
                            options=options, 
                            reduction=reduction, 
                            vectorizer_type=None,
                            ),
                           inputs=[X, y],
                           params={'models': models, 'options': options, 'reduction': reduction, 'vectorizer_type': vectorizer_type}
                        )



    if visualizations:
        viz_dir = 'files/visualizations/fake_news_detection'
        os.makedirs(viz_dir, exist_ok=True)

        plot_model_comparison(result, save_path=os.path.join(viz_dir, 'model_comparison.png'))
        
        plot_confusion_matrices(
            result, 
            save_path=os.path.join(viz_dir, 'confusion_matrices.png')
        )

    return result, preprocessor