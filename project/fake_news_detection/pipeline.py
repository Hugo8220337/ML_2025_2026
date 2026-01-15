import pandas as pd
import numpy as np
from common.nlp import preprocessing as nlp_preprocess
from topic_classification.topic_classification import topic_classification
from anomaly_detection.anomaly_detection import anomaly_detection
from stance_detection.stance_detection import stance_detection
from clickbait_detection.clickbait_detection import clickbait_detection
from tensorflow.keras.preprocessing.sequence import pad_sequences

def _predict_with_pipeline(df, result, text_col, model_name_hint='model'):
    pipeline_data = result.get('pipeline', {})
    model = result.get('model')
    vectorizer = pipeline_data.get('vectorizer')
    reduction_model = pipeline_data.get('reduction_model')
    tokenizer = pipeline_data.get('tokenizer')

    if model is None:
        print(f"Warning: No model found in result for {model_name_hint}. Skipping.")
        return np.zeros(len(df))

    temp_df = pd.DataFrame({'data': df[text_col].fillna('')})
    
    if tokenizer is not None:
        temp_df = nlp_preprocess(temp_df)
        corpus = temp_df['data'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        sequences = tokenizer.texts_to_sequences(corpus)
        try:
            max_len = model.input_shape[1] 
        except:
            max_len = 30
        X_final = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    elif vectorizer:
        temp_df = nlp_preprocess(temp_df)
        corpus = temp_df['data'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        X_vec = vectorizer.transform(corpus)
        if reduction_model:
            X_final = reduction_model.transform(X_vec)
        else:
            X_final = X_vec.toarray() if hasattr(X_vec, 'toarray') else X_vec
    else:
        X_final = temp_df['data']

    try:
        if hasattr(model, 'predict'):
            predictions = model.predict(X_final, verbose=0)
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = (predictions > 0.5).astype(int).flatten()
        elif hasattr(model, 'transform'): 
            W = model.transform(X_final)
            predictions = W.argmax(axis=1)
        else:
            return np.zeros(len(df))
        return predictions
    except Exception as e:
        print(f"Error during prediction for {model_name_hint}: {e}")
        return np.zeros(len(df))

def get_topic_classification_model_predict(df):
    print("Running Topic Classification Predictions...")
    result = topic_classification(visualizations=False)

    df['data'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    preds = _predict_with_pipeline(df, result, 'data', 'Topic Classification')
    df['topic'] = preds
    
    df.drop(columns=['data'], inplace=True)
    return df

def get_anomaly_detection_model_predict(df):
    print("Running Anomaly Detection Predictions...")
    result = anomaly_detection(visualizations=False)
    
    preds = _predict_with_pipeline(df, result, 'text', 'Anomaly Detection')
    df['anomaly'] = preds

    return df

def get_stance_detection_model_predict(df):
    print("Running Stance Detection Predictions...")
    result = stance_detection(visualizations=False)

    df['data'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    preds = _predict_with_pipeline(df, result, 'data', 'Stance Detection')
    df['stance'] = preds
    
    df.drop(columns=['data'], inplace=True)

    return df

def get_clickbait_detection_model_predict(df):
    print("Running Clickbait Detection Predictions...")
    result = clickbait_detection(visualizations=False)
    
    preds = _predict_with_pipeline(df, result, 'title', 'Clickbait Detection')
    df['clickbait'] = preds
    
    return df

def run_pipeline(df):
    df = get_topic_classification_model_predict(df)
    df = get_anomaly_detection_model_predict(df)
    df = get_stance_detection_model_predict(df)
    df = get_clickbait_detection_model_predict(df)
    return df