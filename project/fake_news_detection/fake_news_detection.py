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


def fake_news_detection(models=['logistic_regression'], target_metric='accuracy', reduction=None, options='quick', vectorizer_type='tfidf', visualizations=False):
    cache = CacheManager(module_name='fake_news_detection')
    file_path = 'datasets/FakeNewsNet/'

    

    def preprocessing(file_path):
        Buzzfeed_real_df = read_csv(file_path+'BuzzFeed_real_news_content.csv', usecols=['title', 'text', 'authors', 'source'])
        Buzzfeed_fake_df = read_csv(file_path+'BuzzFeed_fake_news_content.csv', usecols=['title', 'text', 'authors', 'source'])
        PolitiFact_real_df = read_csv(file_path+'PolitiFact_real_news_content.csv', usecols=['title', 'text', 'authors', 'source'])
        PolitiFact_fake_df = read_csv(file_path+'PolitiFact_fake_news_content.csv', usecols=['title', 'text', 'authors', 'source']) 
        
        Buzzfeed_real_df['label'] = 0
        Buzzfeed_fake_df['label'] = 1
        PolitiFact_real_df['label'] = 0
        PolitiFact_fake_df['label'] = 1
        df = pd.concat([Buzzfeed_real_df, Buzzfeed_fake_df, PolitiFact_real_df, PolitiFact_fake_df])
        df = df.sample(frac=1).reset_index(drop=True)


        return df



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)

    # X = df['']
    # y = df['']


    # result = cache.execute(task_name='ems',
    #                        func=lambda: ems(
    #                         X=X,
    #                         y=y, 
    #                         models=models,
    #                         target_metric=target_metric,
    #                         report=True, 
    #                         options=options, 
    #                         reduction=reduction, 
    #                         vectorizer_type=vectorizer_type,
    #                         ),
    #                        inputs=[X, y],
    #                        params={'models': models, 'options': options, 'reduction': reduction, 'vectorizer_type': vectorizer_type})



    # if visualizations:
    #     viz_dir = 'files/visualizations/fake_news_detection'
    #     os.makedirs(viz_dir, exist_ok=True)

    #     plot_model_comparison(result, save_path=os.path.join(viz_dir, 'model_comparison.png'))
        
    #     plot_confusion_matrices(
    #         result, 
    #         save_path=os.path.join(viz_dir, 'confusion_matrices.png')
    #     )