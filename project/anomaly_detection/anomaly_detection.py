import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common.nlp import tfidf_vectorize
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager
from common.visualizations import plot_anomaly_confusion_matrix, plot_anomaly_scatter, plot_model_comparison, plot_clusters


def anomaly_detection(models=['isolation_forest'], target_metric='f1_weighted', reduction=None, options='default', vectorizer_type='tfidf', visualizations=False):
    cache = CacheManager(module_name='anomaly_detection')
    file_path = 'datasets/ISOT/'

    def preprocessing(file_path):
        true_df = read_csv(file_path+'True.csv')
        fake_df = read_csv(file_path+'Fake.csv')
        true_df['label'] = 0
        fake_df['label'] = 1
        true_df['text'] = true_df['text'].str.replace(r'^.*?\(Reuters\) - ', '', regex=True)
        fake_sample = fake_df.sample(frac=0.05, random_state=42)
        df = pd.concat([true_df, fake_sample], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)

    
    X = df['text']
    y = df['label']


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




    if visualizations:
        viz_dir = 'files/visualizations/anomaly_detection'
        os.makedirs(viz_dir, exist_ok=True)

        
        X_tfidf, _ = tfidf_vectorize(df, col_name='text', max_features=5000)   
        _, X_test, _, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
        models_data = result.get('other', {})
        
        for model_name, model_info in models_data.items():
            data_container = model_info.get('info', model_info)

            if 'test_data' not in data_container or data_container['test_data'] is None:
                if 'model' in model_info: 
                    model = model_info['model']
                    try:
                        X_input = X_test
                        if 'autoencoder' in model_name and hasattr(X_test, 'toarray'):
                            X_input = X_test.toarray()
                        
                        if 'dense_autoencoder' in model_name or 'lstm' in model_name:
                            reconstructions = model.predict(X_input, verbose=0)
                            mse = np.mean(np.power(X_input - reconstructions, 2), axis=1)
                            
                            if hasattr(model, 'threshold_'):
                                threshold = model.threshold_
                                preds = (mse > threshold).astype(int)
                        elif 'isolation_forest' in model_name:
                            raw_preds = model.predict(X_input)
                            preds = np.where(raw_preds == -1, 1, 0)
                        else:
                            preds = model.predict(X_input)
                        
                        model_info['test_data'] = {
                            'y_test': y_test,
                            'predictions': preds
                        }
                        
                    except Exception as e:
                        print(f"  -> Failed to predict for {model_name}: {e}")

        plot_anomaly_confusion_matrix(
            result, 
            save_path=os.path.join(viz_dir, 'confusion_matrices.png')
        )

        plot_anomaly_scatter(
            X_test, 
            result, 
            save_path=os.path.join(viz_dir, 'scatter_plot.png')
        )

        plot_model_comparison(
            result, 
            save_path=os.path.join(viz_dir, 'model_comparison.png'),
            title="Anomaly Detection Models (F1-Weighted)"
        )
