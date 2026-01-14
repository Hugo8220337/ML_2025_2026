import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager
from common.visualizations import plot_anomaly_confusion_matrix, plot_anomaly_scatter, plot_model_comparison


def anomaly_detection(
    models=['random_forest', 'svm', 'neural_network'],
    target_metric='f1_score',
    reduction=None,
    options='default',
    vectorizer_type='tfidf',
    visualizations=False,
    type=None
    ):

    cache = CacheManager(module_name='anomaly_detection')
    file_path = 'datasets/ISOT/'

    def preprocessing(file_path):
        true_df = read_csv(file_path+'True.csv')
        fake_df = read_csv(file_path+'Fake.csv')
        true_df['label'] = 0
        fake_df['label'] = 1
        true_df['text'] = true_df['text'].str.replace(r'^.*?\(Reuters\) - ', '', regex=True)
        if type == 'anomaly':
            fake_sample = fake_df.sample(frac=0.05, random_state=42)
            df = pd.concat([true_df, fake_sample], ignore_index=True)
        else:
            df = pd.concat([true_df, fake_df], ignore_index=True)
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

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models_data = result.get('other', {})

        X_final_for_scatter = None 
        
        for model_name, model_result in models_data.items():
            pipeline = model_result.get('pipeline', {})
            vectorizer = pipeline.get('vectorizer')
            reduction_model = pipeline.get('reduction_model')
            model = model_result.get('model')

            if model is None:
                print(f"  [!] No model found for {model_name}. Skipping.")
                continue

            try:
                X_final = None
                
                if 'embedding_autoencoder' in model_name:
                    if hasattr(model, 'tokenizer_'):
                        from tensorflow.keras.preprocessing.sequence import pad_sequences
                        tokenizer = model.tokenizer_
                        test_strs = [" ".join(t) if isinstance(t, list) else str(t) for t in X_test_raw.tolist()]
                        X_final = pad_sequences(tokenizer.texts_to_sequences(test_strs), maxlen=100, padding='post', truncating='post')
                    else:
                        print("  [!] Embedding AE missing tokenizer_. Skipping.")
                        continue
                elif vectorizer:
                    X_vec = vectorizer.transform(X_test_raw)
                    
                    if reduction_model:
                        X_vec = reduction_model.transform(X_vec)
                    
                    if ('dense_autoencoder' in model_name or 'lstm' in model_name) and hasattr(X_vec, 'toarray'):
                        X_final = X_vec.toarray()
                    else:
                        X_final = X_vec
                        
                    X_final_for_scatter = X_final
                else:
                    print(f"  [!] No vectorizer in pipeline for {model_name}.")
                    continue

                preds = None
                
                if 'autoencoder' in model_name or 'lstm' in model_name:
                    if 'embedding' in model_name:
                        rec_preds = model.predict(X_final, verbose=0)
                        epsilon = 1e-7
                        rec_preds = np.clip(rec_preds, epsilon, 1. - epsilon)
                        X_target = tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences(test_strs), mode='binary')
                        bce = - (X_target * np.log(rec_preds) + (1 - X_target) * np.log(1 - rec_preds))
                        error = np.mean(bce, axis=1)
                    else:
                        reconstructions = model.predict(X_final, verbose=0)
                        error = np.mean(np.power(X_final - reconstructions, 2), axis=1)
                    
                    if hasattr(model, 'threshold_'):
                        preds = (error > model.threshold_).astype(int)
                    else:
                        print(f"  [!] No threshold_ found for {model_name}, using fallback.")
                        preds = (error > np.percentile(error, 95)).astype(int)

                elif 'isolation_forest' in model_name or 'one_class_svm' in model_name:
                    raw_preds = model.predict(X_final)
                    preds = np.where(raw_preds == -1, 1, 0)
                else:
                    preds = model.predict(X_final)

                model_result['test_data'] = {
                    'y_test': y_test.values,
                    'predictions': preds
                }

            except Exception as e:
                print(f"  [!] Error predicting {model_name}: {e}")

        plot_anomaly_confusion_matrix(
            result, 
            save_path=os.path.join(viz_dir, 'confusion_matrices.png')
        )

        if X_final_for_scatter is not None:
            plot_anomaly_scatter(
                X_final_for_scatter,
                result, 
                save_path=os.path.join(viz_dir, 'scatter_plot.png')
            )

        plot_model_comparison(
            result, 
            save_path=os.path.join(viz_dir, 'model_comparison.png'),
            title="Anomaly Detection Models (F1-Binary)"
        )

    return result
