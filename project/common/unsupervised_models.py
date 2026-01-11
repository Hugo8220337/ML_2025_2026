import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from .metrics import get_clustering_metrics



def train_kmeans(
    X,
    feature_columns=None,
    n_clusters=3,
    init='k-means++',
    n_init='auto',
    max_iter=300,
    tol=1e-4,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm='lloyd',
    **kwargs
):
    if isinstance(X, pd.DataFrame):
        if feature_columns:
            missing = [col for col in feature_columns if col not in X.columns]
            if missing:
                print(f"Error: Features {missing} not found in DataFrame.")
                return None
            X_data = X[feature_columns]
        else:
            X_data = X
    else:
        X_data = X

    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm
    )

    try:
        model.fit(X_data)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.labels_
    centroids = model.cluster_centers_

    metrics = get_clustering_metrics(X_data, labels, model)

    return {
        "model": model,
        "labels": labels,
        "centroids": centroids,
        "metrics": metrics,
        "inertia": model.inertia_ 
    }


def train_dbscan(
    X,
    feature_columns=None,
    eps=0.5,
    min_samples=5,
    metric='euclidean',
    metric_params=None,
    algorithm='auto',
    leaf_size=30,
    p=None,
    n_jobs=None,
    **kwargs
):
    if isinstance(X, pd.DataFrame):
        if feature_columns:
            missing = [col for col in feature_columns if col not in X.columns]
            if missing:
                print(f"Error: Features {missing} not found in DataFrame.")
                return None
            X_data = X[feature_columns]
        else:
            X_data = X
    else:
        X_data = X

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs
    )

    try:
        model.fit(X_data)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    metrics = get_clustering_metrics(X_data, labels, model)
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = n_noise

    return {
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "metrics": metrics
    }

def train_hdbscan(
    X,
    feature_columns=None,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    metric='euclidean',
    alpha=1.0,
    algorithm='auto',
    leaf_size=40,
    cluster_selection_method='eom',
    allow_single_cluster=False,
    n_jobs=None,
    **kwargs
):
    if isinstance(X, pd.DataFrame):
        if feature_columns:
            missing = [col for col in feature_columns if col not in X.columns]
            if missing:
                print(f"Error: Features {missing} not found in DataFrame.")
                return None
            X_data = X[feature_columns]
        else:
            X_data = X
    else:
        X_data = X

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        n_jobs=n_jobs
    )

    try:
        model.fit(X_data)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.labels_
    probabilities = model.probabilities_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    try:
        metrics = get_clustering_metrics(X_data, labels, model)
        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = n_noise
    except ValueError as e:
        metrics = {
             'n_clusters': n_clusters,
             'n_noise': n_noise,
             'error': str(e)
        }

    return {
        "model": model,
        "labels": labels,
        "probabilities": probabilities,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "metrics": metrics
    }


def train_gmm(
    X,
    feature_columns=None,
    n_components=1,
    covariance_type='full',
    tol=1e-3,
    reg_covar=1e-6,
    max_iter=100,
    n_init=1,
    init_params='kmeans',
    weights_init=None,
    means_init=None,
    precisions_init=None,
    random_state=None,
    warm_start=False,
    verbose=0,
    verbose_interval=10,
    **kwargs
):
    if isinstance(X, pd.DataFrame):
        if feature_columns:
            missing = [col for col in feature_columns if col not in X.columns]
            if missing:
                print(f"Error: Features {missing} not found in DataFrame.")
                return None
            X_data = X[feature_columns]
        else:
            X_data = X
    else:
        X_data = X

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=tol,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        init_params=init_params,
        weights_init=weights_init,
        means_init=means_init,
        precisions_init=precisions_init,
        random_state=random_state,
        warm_start=warm_start,
        verbose=verbose,
        verbose_interval=verbose_interval
    )

    try:
        model.fit(X_data)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.predict(X_data)
    probabilities = model.predict_proba(X_data)

    metrics = get_clustering_metrics(X_data, labels, model)
    metrics['bic'] = model.bic(X_data)
    metrics['aic'] = model.aic(X_data)

    return {
        "model": model,
        "labels": labels,
        "probabilities": probabilities,
        "means": model.means_,
        "covariances": model.covariances_,
        "weights": model.weights_,
        "metrics": metrics,
        "bic": metrics['bic'],
        "aic": metrics['aic']
    }