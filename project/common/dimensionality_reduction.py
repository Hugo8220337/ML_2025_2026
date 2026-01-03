import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation, TruncatedSVD


def apply_pca(
    X,
    n_components=2,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    n_oversamples=10,
    power_iteration_normalizer='auto',
    random_state=None,
    fail_safe=True,
    **kwargs
):
    if fail_safe:
        _failsafe(X, 'pca')

    if sparse.issparse(X):
        X_data = X.toarray()
    elif isinstance(X, pd.DataFrame):
        X_data = X.values
    else:
        X_data = np.asarray(X)
    
    model = PCA(
        n_components=n_components,
        copy=copy,
        whiten=whiten,
        svd_solver=svd_solver,
        tol=tol,
        iterated_power=iterated_power,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state
    )
    
    try:
        reduced_data = model.fit_transform(X_data)
    except Exception as e:
        print(f"Error during PCA transformation: {e}")
        return None
    
    return {
        "reduced_data": reduced_data,
        "model": model,
        "explained_variance": model.explained_variance_,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "components": model.components_,
        "n_components": model.n_components_
    }


def apply_nmf(
    X,
    n_components=2,
    init=None,
    solver='cd',
    beta_loss='frobenius',
    tol=1e-4,
    max_iter=200,
    random_state=None,
    alpha_W=0.0,
    alpha_H='same',
    l1_ratio=0.0,
    verbose=0,
    shuffle=False,
    fail_safe=True,
    **kwargs
):
    if fail_safe:
        _failsafe(X, 'nmf')
    
    if isinstance(X, pd.DataFrame):
        X_data = X.values
    else:
        X_data = X
    
    model = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        verbose=verbose,
        shuffle=shuffle
    )
    
    try:
        reduced_data = model.fit_transform(X_data)
    except Exception as e:
        print(f"Error during NMF transformation: {e}")
        return None
    
    return {
        "reduced_data": reduced_data,
        "model": model,
        "components": model.components_,
        "reconstruction_err": model.reconstruction_err_,
        "n_components": model.n_components_
    }


def apply_lda(
    X,
    n_components=10,
    doc_topic_prior=None,
    topic_word_prior=None,
    learning_method='batch',
    learning_decay=0.7,
    learning_offset=10.0,
    max_iter=10,
    batch_size=128,
    evaluate_every=-1,
    total_samples=1e6,
    perp_tol=1e-1,
    mean_change_tol=1e-3,
    max_doc_update_iter=100,
    n_jobs=None,
    verbose=0,
    random_state=None,
    fail_safe=True,
    **kwargs
):
    if fail_safe:
        _failsafe(X, 'lda')
    
    if isinstance(X, pd.DataFrame):
        X_data = X.values
    else:
        X_data = X
    
    model = LatentDirichletAllocation(
        n_components=n_components,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
        learning_method=learning_method,
        learning_decay=learning_decay,
        learning_offset=learning_offset,
        max_iter=max_iter,
        batch_size=batch_size,
        evaluate_every=evaluate_every,
        total_samples=total_samples,
        perp_tol=perp_tol,
        mean_change_tol=mean_change_tol,
        max_doc_update_iter=max_doc_update_iter,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )
    
    try:
        reduced_data = model.fit_transform(X_data)
    except Exception as e:
        print(f"Error during LDA transformation: {e}")
        return None
    
    try:
        perplexity = model.perplexity(X_data)
    except Exception:
        perplexity = None
    
    return {
        "reduced_data": reduced_data,
        "model": model,
        "components": model.components_,
        "perplexity": perplexity,
        "log_likelihood": model.score(X_data) if X_data is not None else None,
        "n_components": model.n_components
    }


def apply_lsa(
    X,
    n_components=2,
    algorithm='randomized',
    n_iter=5,
    n_oversamples=10,
    power_iteration_normalizer='auto',
    random_state=None,
    tol=0.0,
    fail_safe=True,
    **kwargs
):
    if fail_safe:
        _failsafe(X, 'lsa')
    
    if isinstance(X, pd.DataFrame):
        X_data = X.values
    else:
        X_data = X
    
    model = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
        tol=tol
    )
    
    try:
        reduced_data = model.fit_transform(X_data)
    except Exception as e:
        print(f"Error during LSA transformation: {e}")
        return None
    
    return {
        "reduced_data": reduced_data,
        "model": model,
        "explained_variance": model.explained_variance_,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "singular_values": model.singular_values_,
        "components": model.components_,
        "n_components": model.n_components
    }



def _failsafe(X, method):
    method = method.lower()
    is_sparse = sparse.issparse(X)
    
    if method == 'pca' and is_sparse:
        msg = ("Sparse matrix passed to standard PCA. "
               "This requires densifying the data (X.toarray()), which may cause a "
               "MemoryError/Crash on large datasets. Use LSA (TruncatedSVD) instead, "
               "or set fail_safe=False to attempt it anyway.")
        raise ValueError(msg)

    if method in ['nmf', 'lda']:
        min_val = X.min() if not is_sparse else (X.data.min() if X.nnz > 0 else 0)
        
        if min_val < 0:
            msg = (f"FAILSAFE TRIGGERED: Negative values detected (min={min_val}). "
                   f"{method.upper()} strictly requires non-negative input. "
                   f"Set fail_safe=False to attempt execution (likely to raise sklearn error).")
            raise ValueError(msg)
    
    return