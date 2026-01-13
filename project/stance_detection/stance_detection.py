import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common.nlp import tfidf_vectorize
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager
from common.visualizations import plot_model_comparison, plot_stance_confusion_matrix


def stance_detection(models=['logistic_regression'], target_metric='accuracy', reduction=None, options='quick', vectorizer_type='tfidf', visualizations=False):
    cache = CacheManager(module_name='stance_detection')
    file_path = 'datasets/fnc1/'

    

    def preprocessing(file_path):
        stances_df = read_csv(file_path+'fnc1_train_stances.csv')
        body_df = read_csv(file_path+'fnc1_train_bodies.csv')
        df = pd.merge(stances_df, body_df, on='Body ID')
        df = df.drop(columns=['Body ID'])
        df['data'] = df['Headline'] + ' ' + df['articleBody']
        df = df.drop(columns=['Headline', 'articleBody'])
        df_unrelated = df[df['Stance'] == 'unrelated']
        df_others = df[df['Stance'] != 'unrelated']
        df_downsample = df_unrelated.sample(n=12000, random_state=42)
        df_balanced = pd.concat([df_downsample, df_others])
        df_balanced = df_balanced.sample(frac=1, random_state=42)


        return df_balanced



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)

    X = df['data']
    y = df['Stance']


    result = cache.execute(task_name='ems',
                           func=lambda: ems(
                            X=X,
                            y=y, 
                            models=models,
                            target_metric=target_metric,
                            report=True, 
                            options=options, 
                            reduction=reduction, 
                            vectorizer_type=vectorizer_type,
                            nlp_options={
                                'max_features': 8000,  
                                'ngram_range': (1, 3),
                            },),
                           inputs=[X, y],
                           params={'models': models, 'options': options, 'reduction': reduction, 'vectorizer_type': vectorizer_type})



    if visualizations:
        viz_dir = 'files/visualizations/stance_detection'
        os.makedirs(viz_dir, exist_ok=True)

        plot_model_comparison(result, save_path=os.path.join(viz_dir, 'model_comparison.png'))
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        best_model = result['model']
        best_name = result['info']['model_name']
        pipeline = result['pipeline']
        
        X_test_vec = pipeline['vectorizer'].transform(X_test)
        
        if pipeline.get('reduction_model'):
            X_test_vec = pipeline['reduction_model'].transform(X_test_vec)
            
        y_pred = best_model.predict(X_test_vec)
        
        plot_stance_confusion_matrix(
            y_test, 
            y_pred, 
            model_name=best_name, 
            save_path=os.path.join(viz_dir, f'confusion_matrix_{best_name}.png')
        )