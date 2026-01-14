import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import nltk


def get_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_weighted': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall_weighted': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, average='binary', zero_division=0), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    
    if y_prob is not None:
        if len(np.unique(y_true)) > 2:
            metrics['auc'] = round(roc_auc_score(y_true, y_prob, multi_class='ovr'), 4)
        else:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                 metrics['auc'] = round(roc_auc_score(y_true, y_prob[:, 1]), 4)
            else:
                 raise ValueError("Probability shape mismatch: y_prob must have 2 columns for binary classification.")
            
    return metrics

def get_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'mse': round(mse, 4),
        'rmse': round(np.sqrt(mse), 4),
        'mae': round(mean_absolute_error(y_true, y_pred), 4),
        'r2': round(r2_score(y_true, y_pred), 4),
    }
    return metrics

def get_clustering_metrics(X, labels, model=None, **kwargs):
    metrics = {}
    if model and hasattr(model, 'inertia_'):
        metrics['inertia'] = round(model.inertia_, 4)
    
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        if X.shape[0] > 10000:
            metrics['silhouette_score'] = round(silhouette_score(X, labels, sample_size=10000), 4)
            metrics['silhouette_note'] = "Calculated on 10k sample"
        else:
            metrics['silhouette_score'] = round(silhouette_score(X, labels), 4)
    else:
        metrics['silhouette_score'] = -1.0
        

    target_metric = kwargs.get('target_metric')
    texts = kwargs.get('texts')
    feature_names = kwargs.get('feature_names')
    dictionary = kwargs.get('dictionary')

    if target_metric == 'coherence' and texts is not None:
         topics = None
         
         if hasattr(model, 'components_'):
             components = model.components_
         elif hasattr(model, 'cluster_centers_'):
             components = model.cluster_centers_
         else:
             components = None

         if components is not None:
             if feature_names is not None:
                topics = []
                for topic_idx, topic in enumerate(components):
                    top_n_indices = topic.argsort()[:-11:-1]
                    top_words = [feature_names[i] for i in top_n_indices]
                    topics.append(top_words)

         if topics:
             metrics['coherence'] = get_coherence_score(topics, texts, dictionary=dictionary, coherence='c_v')
         else:
             metrics['coherence'] = 0.0

    return metrics

def get_coherence_score(topics, texts, dictionary=None, coherence='c_v'):
    if texts and isinstance(texts[0], str):
        try:
             try:
                 nltk.data.find('tokenizers/punkt_tab')
             except LookupError:
                 nltk.download('punkt_tab', quiet=True)
        except ImportError:
             pass
        texts = [word_tokenize(text) for text in texts]

    if dictionary is None:
        dictionary = Dictionary(texts)

    coherence_model = CoherenceModel(
        topics=topics, 
        texts=texts, 
        dictionary=dictionary, 
        coherence=coherence,
        processes=1
    )
    return round(coherence_model.get_coherence(), 4)

def evaluate_model(model, X_test, y_test=None):
    
    if is_classifier(model):
        if y_test is None:
            raise ValueError("y_test is required for classification evaluation")
        
        predictions = model.predict(X_test)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_test)
            except Exception:
                probabilities = None 
        
        metrics = get_classification_metrics(y_test, predictions, probabilities)
        return {"model_type": "Classification", "metrics": metrics}

    
    elif is_regressor(model):
        if y_test is None:
            raise ValueError("y_test is required for regression evaluation")
            
        predictions = model.predict(X_test)
        metrics = get_regression_metrics(y_test, predictions)
        return {"model_type": "Regression", "metrics": metrics}

    
    elif hasattr(model, 'fit_predict') or hasattr(model, 'cluster_centers_') or hasattr(model, 'labels_'):
        if hasattr(model, 'predict'):
            labels = model.predict(X_test)
        else:
            raise TypeError("Model does not support 'predict' on new data. Cannot evaluate X_test.")
            
        metrics = get_clustering_metrics(X_test, labels, model)
        return {"model_type": "Clustering", "metrics": metrics}

    raise ValueError("Could not determine a suitable evaluation path for the given model.")