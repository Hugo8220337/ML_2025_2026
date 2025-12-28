import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from .metrics import get_clustering_metrics



def train_kmeans(
    df,
    feature_columns,
    n_clusters=8,
    init='k-means++',
    n_init='auto',
    max_iter=300,
    tol=1e-4,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm='lloyd'
):
    
    for col in feature_columns:
        if col not in df.columns:
            print(f"Error: Feature '{col}' not found in DataFrame.")
            return None

    X = df[feature_columns]

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
        model.fit(X)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.labels_
    centroids = model.cluster_centers_

    metrics = get_clustering_metrics(X, labels, model)

    return {
        "model": model,
        "labels": labels,
        "centroids": centroids,
        "metrics": metrics,
        "inertia": model.inertia_ 
    }
    

def perform_pca(
    df,
    feature_columns,
    n_components=None,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    random_state=None
):
    for col in feature_columns:
        if col not in df.columns:
            print(f"Error: Feature '{col}' not found in DataFrame.")
            return None

    X = df[feature_columns]

    model = PCA(
        n_components=n_components,
        copy=copy,
        whiten=whiten,
        svd_solver=svd_solver,
        tol=tol,
        iterated_power=iterated_power,
        random_state=random_state
    )

    try:
        transformed_data = model.fit_transform(X)
    except Exception as e:
        print(f"Error during PCA: {e}")
        return None

    num_cols = transformed_data.shape[1]
    new_col_names = [f'PC{i+1}' for i in range(num_cols)]
    df_transformed = pd.DataFrame(transformed_data, columns=new_col_names)

    return {
        "model": model,
        "transformed_df": df_transformed,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "components": model.components_
    }


def train_dbscan(
    df,
    feature_columns,
    eps=0.5,
    min_samples=5,
    metric='euclidean',
    metric_params=None,
    algorithm='auto',
    leaf_size=30,
    p=None,
    n_jobs=None
):
    
    for col in feature_columns:
        if col not in df.columns:
            print(f"Error: Feature '{col}' not found in DataFrame.")
            return None

    X = df[feature_columns]

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
        model.fit(X)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    metrics = get_clustering_metrics(X, labels, model)

    return {
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "metrics": metrics
    }


def train_gmm(
    df,
    feature_columns,
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
    verbose_interval=10
):
    
    for col in feature_columns:
        if col not in df.columns:
            print(f"Error: Feature '{col}' not found in DataFrame.")
            return None

    X = df[feature_columns]

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
        model.fit(X)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    labels = model.predict(X)
    probabilities = model.predict_proba(X)

    metrics = get_clustering_metrics(X, labels, model)

    return {
        "model": model,
        "labels": labels,
        "probabilities": probabilities,
        "means": model.means_,
        "covariances": model.covariances_,
        "weights": model.weights_,
        "metrics": metrics,
        "bic": model.bic(X),
        "aic": model.aic(X)
    }