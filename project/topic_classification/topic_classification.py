import json
import os
import numpy as np
import pandas as pd
import umap
from common.nlp import preprocessing as nlp_preprocess
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager
from common.visualizations import plot_model_comparison, plot_clusters

def topic_classification(
    models=['nmf'],
    target_metric='coherence',
    reduction=None,
    options='default',
    vectorizer_type='tfidf',
    visualizations=False
    ):
    
    cache = CacheManager(module_name='topic_classification')
    file_path = 'datasets/allthenews/all-the-news-2-1.csv'

    def preprocessing(file_path):
        try:
            df = read_csv(file_path, usecols=['title', 'article'])
            df = df.dropna()
            df = df.sample(n=100000, random_state=42)
            df['data'] = df['title'] + ' ' + df['article']
            df = df.drop(columns=['title', 'article'])
            return df
        except Exception:
            return




    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)

    
    X = df['data']


    result = cache.execute(task_name='ems',
                           func=lambda: ems(
                            X, 
                            models=models,
                            target_metric=target_metric,
                            report=True, 
                            options=options, 
                            reduction=reduction, 
                            vectorizer_type=vectorizer_type),
                           inputs=X,
                           params={'models': models, 'options': options, 'reduction': reduction, 'vectorizer_type': vectorizer_type})


    if visualizations:
        viz_dir = 'files/visualizations/topic_classification'
        os.makedirs(viz_dir, exist_ok=True)

        plot_model_comparison(result, save_path=os.path.join(viz_dir, 'model_comparison.png'))
        
        pipeline = result.get('pipeline', {})
        other_models = result.get('other', {})
        
        if pipeline:
            sample_size = min(5000, len(X))
            sample_indices = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
            
            vectorizer = pipeline.get('vectorizer')
            reduction_model = pipeline.get('reduction_model')
            
            if vectorizer is not None:
                df_sample = pd.DataFrame({'data': X_sample})
                df_sample = nlp_preprocess(df_sample)
                corpus = df_sample['data'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
                X_vec = vectorizer.transform(corpus)
                
                if reduction_model is not None:
                    X_reduced = reduction_model.transform(X_vec)
                else:
                    X_reduced = X_vec.toarray() if hasattr(X_vec, 'toarray') else X_vec
                
                for model_name, model_data in other_models.items():
                    try:
                        model = model_data.get('model')
                        labels = None
                        components = None

                        if hasattr(model, 'components_'):
                             components = model.components_
                        elif hasattr(model, 'cluster_centers_'):
                             components = model.cluster_centers_

                        if model_name == 'nmf' or (hasattr(model, 'components_') and not hasattr(model, 'predict')):
                            W = model.transform(X_reduced)
                            labels = W.argmax(axis=1)
                            
                            
                            reducer = umap.UMAP(n_components=2, random_state=42)
                            X_plot = reducer.fit_transform(W)
                            
                        elif hasattr(model, 'predict'):
                            labels = model.predict(X_reduced)
                            X_plot = X_reduced
                        elif hasattr(model, 'fit_predict'):
                            labels = model.fit_predict(X_reduced)
                            X_plot = X_reduced
                        else:
                            print(f"Warning: Cannot get labels for {model_name}")
                            continue
                        
                        if components is not None and vectorizer is not None:
                             feature_names = vectorizer.get_feature_names_out()
                             for topic_idx, topic in enumerate(components):
                                top_n = [feature_names[i] for i in topic.argsort()[:-11:-1]]

                        if labels is not None:
                            plot_clusters(X_plot, labels, model_name, 
                                        save_path=os.path.join(viz_dir, f'{model_name}_clusters.png'))
                    except Exception as e:
                        raise Exception(f"Error validating/visualizing model {model_name}: {e}") from e
    return result