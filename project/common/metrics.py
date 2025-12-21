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

def get_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_weighted': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall_weighted': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
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

def get_clustering_metrics(X, labels, model=None):
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
        raise ValueError("Silhouette score cannot be calculated with only 1 cluster.")
        
    return metrics

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