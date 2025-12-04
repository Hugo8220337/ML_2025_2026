import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



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

    score = None
    if 1 < n_clusters < len(X):
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = "Could not calculate silhouette score"

    return {
        "model": model,
        "labels": labels,
        "centroids": centroids,
        "metrics": {"silhouette_score": score},
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