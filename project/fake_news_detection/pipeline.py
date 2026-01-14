import pandas as pd
import numpy as np
from common.nlp import preprocessing as nlp_preprocess
from topic_classification.topic_classification import topic_classification
from anomaly_detection.anomaly_detection import anomaly_detection
from stance_detection.stance_detection import stance_detection
from clickbait_detection.clickbait_detection import clickbait_detection


def get_topic_classification_model_predict(df):
    print("Running Topic Classification...")
    result = topic_classification()
    model = result['model']
    
    

    return df

def get_anomaly_detection_model_predict(df):
    print("Running Anomaly Detection...")
    result = anomaly_detection()
    model = result['model']
    
    
    return df

def get_stance_detection_model_predict(df):
    print("Running Stance Detection...")
    result = stance_detection()
    model = result['model']
    
    
    return df

def get_clickbait_detection_model_predict(df):
    print("Running Clickbait Detection...")
    result = clickbait_detection()
    model = result['model']
    
    
    return df

def run_pipeline(df):
    df = get_topic_classification_model_predict(df)
    df = get_anomaly_detection_model_predict(df)
    df = get_stance_detection_model_predict(df)
    df = get_clickbait_detection_model_predict(df)
    return df