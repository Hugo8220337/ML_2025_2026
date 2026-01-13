import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evolutionary_model_selection.genetic_algorithm import run_genetic_algorithm
from common.supervised_models import *
from common.unsupervised_models import *
from common.deep_learning import train_dense_autoencoder
from common.dimensionality_reduction import apply_pca, apply_nmf, apply_lda, apply_lsa
from common.nlp import preprocessing, tfidf_vectorize, hashing_vectorize
from common.cache import CacheManager
import datetime

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


SUPERVISED_MODELS = {
    'linear_regression',
    'logistic_regression',
    'neural_network',
    'decision_tree',
    'random_forest',
    'knn',
    'svm',
    'naive_bayes',
    'isolation_forest',
    'dense_autoencoder'
}

UNSUPERVISED_MODELS = {
    'kmeans',
    'dbscan',
    'hdbscan',
    'gmm',
    'nmf',
}

DIMENSIONALITY_REDUCTION = {
    'pca',
    'nmf',
    'lda',
    'lsa',
}


MODEL_REGISTRY = {
    # Supervised
    'linear_regression': train_linear_regression,
    'logistic_regression': train_logistic_regression,
    'neural_network': train_neural_network,
    'decision_tree': train_decision_tree,
    'random_forest': train_random_forest,
    'knn': train_knn,
    'svm': train_svm,
    'naive_bayes': train_naive_bayes,
    'isolation_forest': train_isolation_forest,
    'dense_autoencoder': train_dense_autoencoder,
    # Unsupervised
    'kmeans': train_kmeans,
    'dbscan': train_dbscan,
    'hdbscan': train_hdbscan,
    'gmm': train_gmm,
    'nmf': train_nmf,
}

REDUCTION_REGISTRY = {
    'pca': apply_pca,
    'nmf': apply_nmf,
    'lda': apply_lda,
    'lsa': apply_lsa,
}

MODEL_BOUNDS = {
    'logistic_regression': [
        (-4.0, 4.0),   # C (Regularization): 10^-4 to 10^4 (Log Scale)
        (0, 2),        # penalty: 0=l2, 1=l1 (for saga), 2=none
        (0, 2),        # solver: 0=lbfgs, 1=saga
        (0, 2),        # class_weight: 0=None, 1=balanced
    ],
    'neural_network': [
        (10, 300),     # hidden_layer_size (neurons)
        (0, 300),      # hidden_layer_size_2 (neurons)
        (0, 3),        # activation: 0=relu, 1=tanh, 2=logistic
        (0, 2),        # solver: 0=adam, 1=sgd
        (-5.0, -1.0),  # alpha: 10^-5 to 10^-1 (Log Scale)
        (-4.0, -1.0),  # learning_rate_init: 10^-4 to 10^-1 (Log Scale)
    ],
    'decision_tree': [
        (2, 100),      # max_depth
        (2, 40),       # min_samples_split
        (1, 20),       # min_samples_leaf
        (0, 2),        # criterion: 0=gini, 1=entropy
        (0, 3),        # max_features: 0=sqrt, 1=log2, 2=None
        (0.0, 0.05),   # ccp_alpha
        (0, 2),        # class_weight: 0=None, 1=balanced
    ],
    'random_forest': [
        # n_estimators is fixed to 100 in decode_params to reduce variance
        (2, 100),      # max_depth
        (2, 40),       # min_samples_split
        (1, 20),       # min_samples_leaf
        (0, 2),        # criterion: 0=gini, 1=entropy
        (0, 3),        # max_features: 0=sqrt, 1=log2, 2=None
        (0, 2),        # bootstrap: 0=True, 1=False
        (0.0, 0.05),   # ccp_alpha
        (0, 2),        # class_weight: 0=None, 1=balanced
    ],
    'knn': [
        (1, 50),       # n_neighbors
        (0, 2),        # weights: 0=uniform, 1=distance
        (1, 3),        # p (metric): 1=manhattan, 2=euclidean
    ],
    'svm': [
        (-3.0, 3.0),   # C: 10^-3 to 10^3 (Log Scale)
        (0, 4),        # kernel: 0=linear, 1=rbf, 2=poly, 3=sigmoid
        (-4.0, 1.0),   # gamma (if scalar): 10^-4 to 10^1 (Log Scale)
        (0, 2),        # class_weight: 0=None, 1=balanced
    ],
    'naive_bayes': [
        (0, 2),        # distribution: 0=gaussian, 1=multinomial
        (-11.0, -7.0), # var_smoothing: 10^-11 to 10^-7 (Log Scale)
        (-3.0, 1.0),   # alpha: 10^-3 to 10^1 (Log Scale)
        (0, 2),        # fit_prior: 0=True, 1=False
    ],
    'isolation_forest': [
        (50, 300),     # n_estimators
        (0.01, 0.1),   # contamination
        (0, 2),        # max_features: 0=1.0, 1=0.8, 2=0.5
        (0, 2),        # bootstrap: 0=False, 1=True
    ],
    'dense_autoencoder': [
        (16, 128),     # encoding_dim: Size of the bottleneck
        (-4.0, -2.0),  # learning_rate: 1e-4 to 1e-2
        (0.0, 0.5),    # dropout
        (0, 2),        # batch_size: 0=32, 1=64, 2=128
    ],
    # Unsupervised models
    'kmeans': [
        (2, 20),       # n_clusters
        (0, 2),        # init: 0=k-means++, 1=random
        (100, 500),    # max_iter
        (0, 2),        # algorithm: 0=lloyd, 1=elkan
    ],
    'dbscan': [
        (0.1, 2.0),    # eps
        (2, 20),       # min_samples
        (0, 3),        # metric: 0=euclidean, 1=manhattan, 2=cosine
        (10, 50),      # leaf_size
    ],
    'hdbscan': [
        (2, 50),       # min_cluster_size
        (1, 40),       # min_samples
        (0.0, 0.5),    # cluster_selection_epsilon
        (0, 2),        # metric: 0=euclidean, 1=manhattan
        (0.5, 1.5),    # alpha
    ],
    'gmm': [
        (2, 20),       # n_components
        (0, 4),        # covariance_type: 0=full, 1=tied, 2=diag, 3=spherical
        (50, 300),     # max_iter
        (1, 10),       # n_init
    ],
    'nmf': [
        (2, 20),       # n_components
        (0, 2),        # init: 0=nndsvd, 1=random
        (0, 2),        # solver: 0=cd, 1=mu
        (0.0, 1.0),    # alpha_W
        (0.0, 1.0),    # l1_ratio
    ],
}

def get_default_genes(model_name):
    defaults = {}
    
    if model_name == 'logistic_regression':
        # C (10^0 = 1.0), penalty (0=l2), solver (0=lbfgs), class_weight (0=None)
        defaults['genes'] = [0.0, 0.0, 0.0, 0.0]

    elif model_name == 'neural_network':
        # hidden1=100, hidden2=0 (off), act=0 (relu), solver=0 (adam), alpha=-4.0 (1e-4), lr_init=-3.0 (1e-3)
        defaults['genes'] = [100.0, 0.0, 0.0, 0.0, -4.0, -3.0]

    elif model_name == 'decision_tree':
        # max_depth=100 (approx None), split=2, leaf=1, gini(0), max_features=None(2), ccp=0.0, class_weight=None(0)
        defaults['genes'] = [100.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0]

    elif model_name == 'random_forest':
        # n_estimators is fixed in decode.
        # max_depth=100 (approx None), split=2, leaf=1, gini(0), max_features=sqrt(0), bootstrap=True(0), ccp=0.0, class_weight=None(0)
        defaults['genes'] = [100.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    elif model_name == 'knn':
        # n_neighbors=5, weights=uniform(0), p=2
        defaults['genes'] = [5.0, 0.0, 2.0]

    elif model_name == 'svm':
        # C=1.0 (10^0), kernel=linear(0), gamma=-2.0 (scale approx), class_weight=None(0)
        defaults['genes'] = [0.0, 0.0, -2.0, 0.0] 

    elif model_name == 'naive_bayes':
        # distribution=gaussian(0), var_smoothing=1e-9(-9.0), alpha=1.0(0.0), fit_prior=True(0)
        defaults['genes'] = [0.0, -9.0, 0.0, 0.0]

    elif model_name == 'isolation_forest':
        # n_estimators=100, contamination=0.1, max_features=sqrt(0), bootstrap=False(0),
        defaults['genes'] = [100.0, 0.1, 0.0, 0.0]
        
    elif model_name == 'dense_autoencoder':
        # Default: 32 dim, 1e-3 lr, 0.1 dropout, 64 batch
        defaults['genes'] = [32.0, -3.0, 0.1, 1.0]

    # Unsupervised models
    elif model_name == 'kmeans':
        # n_clusters=3, init=k-means++(0), max_iter=300, algorithm=lloyd(0)
        defaults['genes'] = [3.0, 0.0, 300.0, 0.0]

    elif model_name == 'dbscan':
        # eps=0.5, min_samples=5, metric=euclidean(0), leaf_size=30
        defaults['genes'] = [0.5, 5.0, 0.0, 30.0]

    elif model_name == 'hdbscan':
        # min_cluster_size=5, min_samples=5, epsilon=0.0, metric=euclidean(0), alpha=1.0
        defaults['genes'] = [5.0, 5.0, 0.0, 0.0, 1.0]

    elif model_name == 'gmm':
        # n_components=2, covariance_type=full(0), max_iter=100, n_init=1
        defaults['genes'] = [2.0, 0.0, 100.0, 1.0]

    elif model_name == 'nmf':
        # n_components=2, init=nndsvd(0), solver=cd(0), alpha_W=0.0, l1_ratio=0.0
        defaults['genes'] = [2.0, 0.0, 0.0, 0.0, 0.0]

    return defaults.get('genes', [])

def decode_params(model_name, genes):
    params = {}
    
    if model_name == 'logistic_regression':
        params['C'] = float(10 ** genes[0])
        
        solver_map = {0: 'lbfgs', 1: 'saga'}
        solver = solver_map.get(int(genes[2]), 'lbfgs')
        params['solver'] = solver
        
        raw_penalty = int(genes[1]) 
        
        if solver == 'lbfgs':
            params['penalty'] = 'l2' if raw_penalty == 0 else None
        elif solver == 'saga':
            penalty_map = {0: 'l2', 1: 'l1', 2: None}
            params['penalty'] = penalty_map.get(raw_penalty, 'l2')

        cw_map = {0: None, 1: 'balanced'}
        params['class_weight'] = cw_map.get(int(genes[3]), None)

    elif model_name == 'neural_network':
        h1 = int(genes[0])
        h2 = int(genes[1])
        if h2 < 10:
            params['hidden_layer_sizes'] = (h1,)
        else:
            params['hidden_layer_sizes'] = (h1, h2)
        
        act_map = {0: 'relu', 1: 'tanh', 2: 'logistic'}
        params['activation'] = act_map.get(int(genes[2]), 'relu')
        
        solver_map = {0: 'adam', 1: 'sgd'}
        solver = solver_map.get(int(genes[3]), 'adam')
        params['solver'] = solver
        
        params['alpha'] = float(10 ** genes[4])
        params['learning_rate_init'] = float(10 ** genes[5])
        
        params['max_iter'] = 200
        params['early_stopping'] = False 

    elif model_name == 'decision_tree':
        params['max_depth'] = int(genes[0])
        params['min_samples_split'] = int(genes[1])
        params['min_samples_leaf'] = int(genes[2])
        
        crit_map = {0: 'gini', 1: 'entropy'}
        params['criterion'] = crit_map.get(int(genes[3]), 'gini')
                
        feat_map = {0: 'sqrt', 1: 'log2', 2: None}
        params['max_features'] = feat_map.get(int(genes[4]), None)
        
        params['ccp_alpha'] = float(genes[5])
        
        cw_map = {0: None, 1: 'balanced'}
        params['class_weight'] = cw_map.get(int(genes[6]), None)

    elif model_name == 'random_forest':
        params['n_estimators'] = 100
        
        params['max_depth'] = int(genes[0])
        params['min_samples_split'] = int(genes[1])
        params['min_samples_leaf'] = int(genes[2])
        
        crit_map = {0: 'gini', 1: 'entropy'}
        params['criterion'] = crit_map.get(int(genes[3]), 'gini')
        
        feat_map = {0: 'sqrt', 1: 'log2', 2: None}
        params['max_features'] = feat_map.get(int(genes[4]), 'sqrt')

        params['bootstrap'] = True if int(genes[5]) == 0 else False
        params['ccp_alpha'] = float(genes[6])
        
        cw_map = {0: None, 1: 'balanced'}
        params['class_weight'] = cw_map.get(int(genes[7]), None)

    elif model_name == 'knn':
        params['n_neighbors'] = int(genes[0])
        
        w_map = {0: 'uniform', 1: 'distance'}
        params['weights'] = w_map.get(int(genes[1]), 'uniform')
        
        params['p'] = int(genes[2])

    elif model_name == 'svm':
        params['C'] = float(10 ** genes[0])
        
        k_map = {0: 'linear', 1: 'rbf', 2: 'poly', 3: 'sigmoid'}
        kernel = k_map.get(int(genes[1]), 'linear')
        params['kernel'] = kernel
        
        if kernel in ['rbf', 'poly', 'sigmoid']:
            params['gamma'] = float(10 ** genes[2])
        else:
            params['gamma'] = 'scale'
            
        cw_map = {0: None, 1: 'balanced'}
        params['class_weight'] = cw_map.get(int(genes[3]), None)

    elif model_name == 'naive_bayes':
        dist_map = {0: 'gaussian', 1: 'multinomial'}
        distribution = dist_map.get(int(genes[0]), 'gaussian')
        params['distribution'] = distribution
        
        if distribution == 'gaussian':
            params['var_smoothing'] = float(10 ** genes[1])
        elif distribution == 'multinomial':
            params['alpha'] = float(10 ** genes[2])
            params['fit_prior'] = True if int(genes[3]) == 0 else False

    elif model_name == 'isolation_forest':
        params['n_estimators'] = int(genes[0])
        params['contamination'] = float(genes[1])
        feat_map = {0: 1.0, 1: 0.8, 2: 0.5}
        params['max_features'] = feat_map.get(int(genes[2]), 1.0)
        params['bootstrap'] = True if int(genes[3]) == 1 else False
        params['n_jobs'] = -1

    elif model_name == 'dense_autoencoder':
        params['encoding_dim'] = int(genes[0])
        params['learning_rate'] = float(10 ** genes[1])
        params['dropout'] = float(genes[2])
        batch_map = {0: 32, 1: 64, 2: 128}
        params['batch_size'] = batch_map.get(int(genes[3]), 64)
        params['epochs'] = 5 

    # Unsupervised models
    elif model_name == 'kmeans':
        params['n_clusters'] = int(genes[0])
        
        init_map = {0: 'k-means++', 1: 'random'}
        params['init'] = init_map.get(int(genes[1]), 'k-means++')
        
        params['max_iter'] = int(genes[2])
        
        algo_map = {0: 'lloyd', 1: 'elkan'}
        params['algorithm'] = algo_map.get(int(genes[3]), 'lloyd')

    elif model_name == 'dbscan':
        params['eps'] = float(genes[0])
        params['min_samples'] = int(genes[1])
        
        metric_map = {0: 'euclidean', 1: 'manhattan', 2: 'cosine'}
        params['metric'] = metric_map.get(int(genes[2]), 'euclidean')
        
        params['leaf_size'] = int(genes[3])

    elif model_name == 'hdbscan':
        params['min_cluster_size'] = int(genes[0])
        params['min_samples'] = int(genes[1])
        params['cluster_selection_epsilon'] = float(genes[2])
        
        metric_map = {0: 'euclidean', 1: 'manhattan'}
        params['metric'] = metric_map.get(int(genes[3]), 'euclidean')
        
        params['alpha'] = float(genes[4])

    elif model_name == 'gmm':
        params['n_components'] = int(genes[0])
        
        cov_map = {0: 'full', 1: 'tied', 2: 'diag', 3: 'spherical'}
        params['covariance_type'] = cov_map.get(int(genes[1]), 'full')
        
        params['max_iter'] = int(genes[2])
        params['n_init'] = int(genes[3])

    elif model_name == 'nmf':
        params['n_components'] = int(genes[0])
        
        init_map = {0: 'nndsvd', 1: 'random'}
        params['init'] = init_map.get(int(genes[1]), 'nndsvd')
        
        solver_map = {0: 'cd', 1: 'mu'}
        params['solver'] = solver_map.get(int(genes[2]), 'cd')
        
        params['alpha_W'] = float(genes[3])
        params['l1_ratio'] = float(genes[4])

    return params

def reduce_dimensions(X, method, **kwargs):
    method = method.lower()
    
    if method not in DIMENSIONALITY_REDUCTION:
        raise ValueError(
            f"Unknown dimensionality reduction method: '{method}'. "
            f"Available methods: {list(DIMENSIONALITY_REDUCTION)}"
        )
    
    return REDUCTION_REGISTRY[method](X, **kwargs)


def preprocess_text(X_train, X_test=None, vectorizer_type='tfidf', **kwargs):
    is_text = False
    if isinstance(X_train, pd.Series) and X_train.dtype == 'object':
        is_text = True
    elif isinstance(X_train, pd.DataFrame) and X_train.shape[1] == 1 and X_train.iloc[:, 0].dtype == 'object':
        is_text = True
    
    if not is_text:
        return X_train, X_test, None
    prep_keys = ['to_lower', 'remove_punctuation', 'remove_digits', 'tokenize', 'remove_stopwords', 'lemmatize']
    vect_keys = ['max_features', 'lowercase', 'stop_words', 'ngram_range']
    
    prep_kwargs = {k: v for k, v in kwargs.items() if k in prep_keys}
    vect_kwargs = {k: v for k, v in kwargs.items() if k in vect_keys}

    cm = CacheManager(module_name="ems")

    def _run_preprocessing():
        def to_df(data):
            if isinstance(data, pd.Series):
                return data.to_frame(name='data')
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
                df.columns = ['data']
                return df
            return None

        df_train = to_df(X_train)
        df_train = preprocessing(df_train, **prep_kwargs)

        tokens_train = df_train['data'].tolist()

        if vectorizer_type == 'hashing':
            X_train_vec, vectorizer = hashing_vectorize(df_train, col_name='data', **vect_kwargs)
        else:
            X_train_vec, vectorizer = tfidf_vectorize(df_train, col_name='data', **vect_kwargs)
        
        X_test_vec = None
        tokens_test = None
        if X_test is not None:
            df_test = to_df(X_test)
            df_test = preprocessing(df_test, **prep_kwargs)
            tokens_test = df_test['data'].tolist()
            corpus_test = df_test['data'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
            X_test_vec = vectorizer.transform(corpus_test)
            
        return {
            'X_train_vec': X_train_vec, 
            'X_test_vec': X_test_vec, 
            'vectorizer': vectorizer, 
            'tokens_train': tokens_train, 
            'tokens_test': tokens_test
        }

    return cm.execute(
        task_name="text_preprocessing",
        func=_run_preprocessing,
        inputs={'X_train': X_train, 'X_test': X_test},
        params=kwargs
    )

def _get_options_key(options):
    if isinstance(options, str):
        return options.lower()
    elif isinstance(options, dict):
        return str(sorted(options.items()))
    else:
        return 'default'


def ems(X, y=None, models=None, reduction=None, target_metric=None, report=False, options=None, nlp_options=None, fulldatatrain=True, vectorizer_type='tfidf'):
    is_unsupervised = y is None
    models_validation(models, is_unsupervised)
    if target_metric is None:
        target_metric = 'silhouette_score' if is_unsupervised else 'accuracy'

    minimize_metrics = {'inertia', 'mse', 'mae', 'rmse', 'bic', 'aic'}
    should_maximize = target_metric not in minimize_metrics

    presets = {
        # Around 50 models trained
        'quick': {
            'population_size': 10,
            'generations': 5,
            'mutation_rate': 0.3,
            'crossover_rate': 0.8,
            'tournament_size': 3,
            'maximize': should_maximize,
            'verbose': True,
            'patience': 3,
            'min_delta': 0.001,
            'elitism_count': 1,
            'mutation_strength': 0.3,
        },
        # Around 375 models trained
        'default': {
            'population_size': 15,
            'generations': 25,
            'mutation_rate': 0.3,
            'crossover_rate': 0.8,
            'tournament_size': 3,
            'maximize': should_maximize,
            'verbose': True,
            'patience': 10,
            'min_delta': 0.0001,
            'elitism_count': 1,
            'mutation_strength': 0.3,
        },
        # Around 2500 models trained
        'slow': {
            'population_size': 25,
            'generations': 100,
            'mutation_rate': 0.3,
            'crossover_rate': 0.8,
            'tournament_size': 5,
            'maximize': should_maximize,
            'verbose': True,
            'patience': 20,
            'min_delta': 0.00001,
            'elitism_count': 2,
            'mutation_strength': 0.2,
        }
    }

    nlp_params = nlp_options if nlp_options is not None else {}
    options_key = _get_options_key(options)
    
    model_cache = CacheManager(module_name="ems_models")
    
    if isinstance(options, str):
        run_options = presets.get(options.lower(), presets['default'])
    elif isinstance(options, dict):
        run_options = presets['default'].copy()
        run_options.update(options)
    else:
        run_options = presets['default']

    best_global_model = None
    best_global_info = {}
    best_global_pipeline = {}
    all_models_results = {}
    candidates = []

    if report:
        reports_dir = "files/ga_logs"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)

        report_file = os.path.join(reports_dir, f"evolution_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_data = []

    n_samples = X.shape[0]
    max_fitness_samples = min(5000, int(n_samples * 0.1))
    
    if is_unsupervised:
        if n_samples > max_fitness_samples:
            print(f"-> Using {max_fitness_samples} samples for evaluation (full: {n_samples})")
            sys.stdout.flush()
            indices = np.random.RandomState(42).choice(n_samples, max_fitness_samples, replace=False)
            X_sub = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        else:
            X_sub = X
        
        preproc_result = preprocess_text(X_sub, vectorizer_type=vectorizer_type, **nlp_params)
        X_train_processed = preproc_result['X_train_vec']
        vectorizer = preproc_result['vectorizer']
        tokens_train = preproc_result['tokens_train']
        
        if reduction is not None:
            print(f"-> Applying dimensionality reduction: {reduction}")
            reduction_result = reduce_dimensions(X_train_processed, method=reduction)
            X_train_processed = reduction_result['reduced_data']
        
        X_test_processed = None
        y_train, y_test = None, None
    else:
        if n_samples > max_fitness_samples:
            print(f"-> Using {max_fitness_samples} samples for evaluation (full: {n_samples})")
            sys.stdout.flush()
            
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=max_fitness_samples, random_state=42
            )
        else:
            X_sub, y_sub = X, y

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
        
        preproc_result = preprocess_text(X_train, X_test, vectorizer_type=vectorizer_type, **nlp_params)
        X_train_processed = preproc_result['X_train_vec']
        X_test_processed = preproc_result['X_test_vec']
        vectorizer = preproc_result['vectorizer']
        tokens_train = preproc_result['tokens_train']
        if reduction is not None:
            print(f"-> Applying dimensionality reduction: {reduction}")
            reduction_result = reduce_dimensions(X_train_processed, method=reduction)
            X_train_processed = reduction_result['reduced_data']
            if X_test_processed is not None:
                X_test_to_transform = X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed
                try:
                    X_test_processed = reduction_result['model'].transform(X_test_to_transform)
                except Exception as e:
                    print(f"-> Warning: Failed to transform test data with reduction model: {e}")
                    raise e

    X_final_train = X
    y_final_train = y
    
    if not fulldatatrain:
        final_sample_size = min(20000, int(n_samples * 0.25))
        if n_samples > final_sample_size:
            print(f"-> Using {final_sample_size} samples for final training (full: {n_samples})")
            indices = np.random.RandomState(42).choice(n_samples, final_sample_size, replace=False)
            
            X_final_train = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            if y is not None:
                y_final_train = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]


    for model_name in models:
        cache_task_name = model_name
        cache_inputs = {'X': X, 'y': y, 'nlp_params': nlp_params, 'target_metric': target_metric}
        
        cached_result = model_cache.execute(
            task_name=cache_task_name,
            func=lambda: None,  
            inputs=cache_inputs,
            params={'options_key': options_key},
            strategy='load_only'
        )
        
        if cached_result is not None and isinstance(cached_result, dict) and 'score' in cached_result:
            candidates.append({
                'model_name': model_name,
                'score': cached_result['score'],
                'params': cached_result['info']['params'],
                'is_cached': True,
                'cached_result': cached_result
            })

            all_models_results[model_name] = {
                'model': cached_result['model'],
                'score': cached_result['score'],
                'info': cached_result['info'],
                'pipeline': cached_result.get('pipeline', {})
            }
            continue 
        
        print(f"Training {model_name}...")
        
        train_func = MODEL_REGISTRY.get(model_name)
        bounds = MODEL_BOUNDS.get(model_name, [])

        if bounds:
            failed_evaluations = [0]
            eval_counter = [0]
            
            print(f"   -> Evaluating default parameters for {model_name}...")
            
            worst_score = -float('inf') if should_maximize else float('inf')
            
            feature_names = None
            if hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
            
            try:
                default_result = train_func(X_train_processed, y_train, X_test=X_test_processed, y_test=y_test, target_metric=target_metric, texts=tokens_train, feature_names=feature_names)
                if default_result and 'metrics' in default_result:
                    default_score = default_result['metrics'].get(target_metric, worst_score)
                    print(f"   -> Default Score ({target_metric}): {default_score:.4f}")
                else:
                    print("   -> Default training failed or returned no metrics.")
                    default_score = worst_score
            except Exception as e:
                print(f"   -> Error training defaults: {e}")
                default_score = worst_score
            
            def fitness_function(individual):
                eval_counter[0] += 1
                params = decode_params(model_name, individual)
                
                try:
                    result = train_func(X_train_processed, y_train, X_test=X_test_processed, y_test=y_test, target_metric=target_metric, texts=tokens_train, feature_names=feature_names, **params)
                    if result and 'metrics' in result:
                        score = result['metrics'].get(target_metric, worst_score)
                        print(f"   [Eval {eval_counter[0]}] Score: {score:.4f}", end='\r')
                        sys.stdout.flush()
                        return score
                except Exception as e:
                    failed_evaluations[0] += 1
                    if failed_evaluations[0] <= 3:
                        print(f"   [DEBUG] Fitness eval failed for {model_name}: {e}")
                
                return worst_score

            def generation_report(gen, scores, population):
                if should_maximize:
                    best_idx = int(np.argmax(scores))
                else:
                    best_idx = int(np.argmin(scores))

                best_score = float(scores[best_idx])
                best_genes = population[best_idx]
                best_params = decode_params(model_name, best_genes)
                
                valid_scores = scores[scores != worst_score]
                
                generation_data = {
                    "model": model_name,
                    "generation": gen,
                    "stats": {
                        "best_score": best_score,
                        "avg_score": float(np.mean(valid_scores)) if len(valid_scores) > 0 else worst_score,
                        "min_score": float(np.min(valid_scores)) if len(valid_scores) > 0 else None,
                        "max_score": float(np.max(valid_scores)) if len(valid_scores) > 0 else None,
                        "failed_evals": int(np.sum(scores == worst_score))
                    },
                    "best": {
                        "index": best_idx,
                        "score": best_score,
                        "parameters": best_params
                    },
                    "individuals": []
                }
                
                for i, (score, genes) in enumerate(zip(scores, population)):
                    readable_params = decode_params(model_name, genes)
                    generation_data["individuals"].append({
                        "index": i,
                        "score": float(score),
                        "parameters": readable_params
                    })
                
                if report:
                    report_data.append(generation_data)

            seed_genes = get_default_genes(model_name)
            seeds = []
            if seed_genes:
                seeds.append(seed_genes)  
                import random
                for _ in range(2):
                    varied = []
                    for i, g in enumerate(seed_genes):
                        low, high = bounds[i]
                        variation = random.uniform(-0.3, 0.3) * (high - low)
                        new_val = max(low, min(high, g + variation))
                        varied.append(new_val)
                    seeds.append(varied)

            ga_result = run_genetic_algorithm(
                fitness_function=fitness_function,
                gene_bounds=bounds,
                seeds=seeds if seeds else None,
                population_size=run_options['population_size'],
                generations=run_options['generations'],
                mutation_rate=run_options['mutation_rate'],
                crossover_rate=run_options['crossover_rate'],
                tournament_size=run_options['tournament_size'],
                maximize=run_options['maximize'],
                verbose=run_options['verbose'],
                patience=run_options['patience'],
                min_delta=run_options['min_delta'],
                elitism_count=run_options['elitism_count'],
                mutation_strength=run_options['mutation_strength'],
                generation_report=generation_report,
            )

            print(f"\n   -> GA Best Score: {ga_result['best_score']:.4f} vs Default: {default_score:.4f}")
            
            improved = ga_result['best_score'] > default_score if should_maximize else ga_result['best_score'] < default_score
            
            final_train_params = {}
            best_run_score = ga_result['best_score'] if improved else default_score

            if improved:
                print(f"   -> Model improved...")
                final_train_params = decode_params(model_name, ga_result['best_solution'])
            else:
                print(f"   -> Default params are optimal. Keeping defaults for final selection...")
                final_train_params = {} 

            candidates.append({
                'model_name': model_name,
                'score': best_run_score,
                'params': final_train_params,
                'is_cached': False
            })

            try:
                partial_result = train_func(X_train_processed, y_train, X_test=X_test_processed, y_test=y_test, target_metric=target_metric, texts=tokens_train, feature_names=feature_names, **final_train_params)
                partial_model = partial_result.get('model')
                partial_score = partial_result['metrics'].get(target_metric, best_run_score)
                
                all_models_results[model_name] = {
                    'model': partial_model,
                    'score': partial_score,
                    'info': {'model_name': model_name, 'score': partial_score, 'params': final_train_params},
                    'pipeline': {}  
                }
            except Exception as e:
                print(f"   -> Error storing partial model for {model_name}: {e}")
                all_models_results[model_name] = {
                    'model': None,
                    'info': {'score': best_run_score, 'params': final_train_params},
                    'pipeline': {}
                }

        else:
            print(f"   -> Running default training for {model_name}...")
            try:
                result = train_func(X_train_processed, y_train, X_test=X_test_processed, y_test=y_test, target_metric=target_metric, texts=tokens_train, feature_names=feature_names)
                score = result['metrics'].get(target_metric, -float('inf') if should_maximize else float('inf'))
                print(f"   -> Score: {score:.4f}")
                
                candidates.append({
                    'model_name': model_name,
                    'score': score,
                    'params': {},
                    'is_cached': False
                })
                
                all_models_results[model_name] = {
                    'model': result.get('model'),
                    'score': score,
                    'info': {'model_name': model_name, 'score': score, 'params': {}},
                    'pipeline': {}      
                }
            except Exception as e:
                print(f"   -> Error evaluating {model_name}: {e}")


    if not candidates:
        raise ValueError("No successful model evaluations.")
    

    if should_maximize:
        best_candidate = max(candidates, key=lambda x: x['score'])
    else:
        best_candidate = min(candidates, key=lambda x: x['score'])
    
    print(f"\n-> Best model selected: {best_candidate['model_name']} (Score: {best_candidate['score']:.4f})")
    
    if best_candidate['is_cached']:
        print("   -> Using cached model.")
        cached = best_candidate['cached_result']
        best_global_model = cached['model']
        best_global_info = cached['info']
        best_global_pipeline = cached.get('pipeline', {})
    
    else:
        model_name = best_candidate['model_name']
        print(f"   -> Training final {model_name} with full data...")
        
        cache_task_name = model_name
        cache_inputs = {'X': X, 'y': y, 'nlp_params': nlp_params, 'target_metric': target_metric}
        
        train_func = MODEL_REGISTRY.get(model_name)
        final_train_params = best_candidate['params']
        final_params_info = final_train_params if final_train_params else 'default'

        try:
            preproc_result = preprocess_text(X_final_train, vectorizer_type=vectorizer_type, **nlp_params)
            X_final = preproc_result['X_train_vec']
            final_vectorizer = preproc_result['vectorizer']
            final_tokens = preproc_result['tokens_train']
            
            final_reduction_model = None
            if reduction is not None:
                reduction_result = reduce_dimensions(X_final, method=reduction)
                X_final = reduction_result['reduced_data']
                final_reduction_model = reduction_result['model']

            run_inputs = [X_final]
            if y is not None:
                run_inputs.append(y_final_train)
            
            final_feature_names = None
            if hasattr(final_vectorizer, 'get_feature_names_out'):
                final_feature_names = final_vectorizer.get_feature_names_out()

            final_run = train_func(*run_inputs, target_metric=target_metric, texts=final_tokens, feature_names=final_feature_names, **final_train_params)
            
            final_score = final_run['metrics'].get(target_metric, best_candidate['score'])
            print(f"   -> Final Score: {final_score:.4f}")

            model_info = {
                'model_name': model_name,
                'score': final_score,
                'params': final_params_info
            }
            
            pipeline_dict = {
                'vectorizer': final_vectorizer,
                'reduction_model': final_reduction_model,
                'classifier': final_run['model']
            }
            
            cache_result = {
                'model': final_run['model'], 
                'score': final_score, 
                'info': model_info,
                'pipeline': pipeline_dict
            }
            
            model_cache.execute(
                task_name=cache_task_name,
                func=lambda r=cache_result: r,
                inputs=cache_inputs,
                params={'options_key': options_key},
            )
            
            best_global_model = final_run['model']
            best_global_info = model_info
            best_global_pipeline = pipeline_dict
            
            all_models_results[model_name] = cache_result

        except Exception as e:
            print(f"   -> Error during final training of {model_name}: {e}")
            pass

    if report and report_data:
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

    return {
        "model": best_global_model,
        "info": best_global_info,
        "pipeline": best_global_pipeline,
        "other": all_models_results
    }


def models_validation(models, is_unsupervised=False):
    if not models:
        raise ValueError("Models list cannot be empty.")
    
    if not all(model in MODEL_REGISTRY for model in models):
        invalid = [m for m in models if m not in MODEL_REGISTRY]
        raise ValueError(f"Unknown model(s): {invalid}. Valid models: {list(MODEL_REGISTRY.keys())}")
    
    has_supervised = any(model in SUPERVISED_MODELS for model in models)
    has_unsupervised = any(model in UNSUPERVISED_MODELS for model in models)
    
    if has_supervised and has_unsupervised:
        raise ValueError(
            "Cannot mix supervised and unsupervised models. "
            f"Supervised: {[m for m in models if m in SUPERVISED_MODELS]}, "
            f"Unsupervised: {[m for m in models if m in UNSUPERVISED_MODELS]}"
        )
    
    if is_unsupervised and has_supervised:
        raise ValueError(
            f"y=None indicates unsupervised mode, but supervised models were provided: "
            f"{[m for m in models if m in SUPERVISED_MODELS]}. "
            f"Use unsupervised models instead: {list(UNSUPERVISED_MODELS)}"
        )
    
    if not is_unsupervised and has_unsupervised:
        raise ValueError(
            f"y is provided (supervised mode), but unsupervised models were provided: "
            f"{[m for m in models if m in UNSUPERVISED_MODELS]}. "
            f"Use supervised models instead: {list(SUPERVISED_MODELS)}"
        )
    

