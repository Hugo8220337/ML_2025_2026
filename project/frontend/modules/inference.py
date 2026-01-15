import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from topic_classification.topic_classification import topic_classification
from anomaly_detection.anomaly_detection import anomaly_detection
from stance_detection.stance_detection import stance_detection
from clickbait_detection.clickbait_detection import clickbait_detection
from fake_news_detection.fake_news_detection import fake_news_detection 
from common.nlp import preprocessing as nlp_preprocess


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _predict_with_pipeline(df, result, text_col, model_name_hint='model'):
    pipeline_data = result.get('pipeline', {})
    model = result.get('model')
    
    if model is None:
        print(f"[Inference] Warning: No main model found for {model_name_hint}.")
        return np.zeros(len(df))

    vectorizer = pipeline_data.get('vectorizer')
    reduction_model = pipeline_data.get('reduction_model')
    tokenizer = pipeline_data.get('tokenizer')
    
    temp_df = pd.DataFrame({'data': df[text_col].fillna('').astype(str)})
    
    if tokenizer is not None:
        temp_df = nlp_preprocess(temp_df)
        corpus = temp_df['data'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        sequences = tokenizer.texts_to_sequences(corpus)
        max_len = model.input_shape[1] if hasattr(model, 'input_shape') else 30
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
            preds = model.predict(X_final)
            if len(preds.shape) > 1 and preds.shape[1] == 1:
                preds = (preds > 0.5).astype(int).flatten()
            
            if 'Anomaly' in model_name_hint and -1 in preds:
                 preds = np.where(preds == -1, 1, 0)
            return preds
        elif hasattr(model, 'transform'): 
            W = model.transform(X_final)
            return W.argmax(axis=1)
    except Exception as e:
        print(f"Error in {model_name_hint}: {e}")
        return np.zeros(len(df))
    
    return np.zeros(len(df))




def get_topic_prediction(df):
    print("Running Topic Classification...")
    result = topic_classification(visualizations=False) 
    
    df['data_topic'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    preds = _predict_with_pipeline(df, result, 'data_topic', 'Topic')
    
    df['topic'] = preds.astype(int)
    df.drop(columns=['data_topic'], inplace=True)
    return df

def get_anomaly_prediction(df):
    print("Running Anomaly Detection...")
    result = anomaly_detection(visualizations=False)
    
    preds = _predict_with_pipeline(df, result, 'text', 'Anomaly')
    df['anomaly'] = preds.astype(int)
    return df

def get_stance_prediction(df):
    print("Running Stance Detection...")
    result = stance_detection(visualizations=False)
    
    df['data_stance'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    preds = _predict_with_pipeline(df, result, 'data_stance', 'Stance')

    df['stance'] = preds 
    df.drop(columns=['data_stance'], inplace=True)
    return df

def get_clickbait_prediction(df):
    print("Running Clickbait Detection...")
    result = clickbait_detection(visualizations=False)
    
    preds = _predict_with_pipeline(df, result, 'title', 'Clickbait')
    df['clickbait'] = preds.astype(int)
    return df


def get_fake_news_prediction(df):
    print("Running Fake News Detection...")
    
    result, preprocessor = fake_news_detection(visualizations=False)
    model = result.get('model')
    pipeline = result.get('pipeline', {})
    reduction_model = pipeline.get('reduction_model')
    
    if model is None:
        raise ValueError("Fake News model not found. Check training cache.")

    X_features = pd.DataFrame()
    X_features['data'] = df['title'].fillna('') + " " + df['text'].fillna('')
    X_features['topic'] = df['topic']
    X_features['stance'] = df['stance']
    X_features['anomaly'] = df['anomaly']
    X_features['clickbait'] = df['clickbait']

    X_final = preprocessor.transform(X_features)
    
    if reduction_model is not None:
        try:
            X_final = reduction_model.transform(X_final)
        except Exception as e:
            if hasattr(X_final, 'toarray'):
                X_final = reduction_model.transform(X_final.toarray())
            else:
                raise e

    pred = model.predict(X_final)[0]
    
    confidence = 1.0
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_final)[0]
        confidence = probs[0] if pred == 0 else probs[1]

    label = "Fake" if pred == 1 else "Real"
    
    df['final_prediction'] = label
    df['confidence'] = confidence
    
    return df


def predict(title, text):
    df = pd.DataFrame({'title': [title], 'text': [text]})

    df = get_topic_prediction(df)
    df = get_anomaly_prediction(df)
    df = get_stance_prediction(df)
    df = get_clickbait_prediction(df)

    df = get_fake_news_prediction(df)

    return df