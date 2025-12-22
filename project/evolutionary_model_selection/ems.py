import os
from evolutionary_model_selection.genetic_algorithm import run_genetic_algorithm
from common.supervised_models import *
from common.unsupervised_models import *
from common.deep_learning import *
import datetime


function_registry = {
    # Supervised
    'linear_regression': train_linear_regression,
    'logistic_regression': train_logistic_regression,
    'neural_network': train_neural_network,
    'decision_tree': train_decision_tree,
    'random_forest': train_random_forest,
    'knn': train_knn,
    'svm': train_svm,
    # Unsupervised
    'kmeans': train_kmeans,
    'pca': perform_pca
}

MODEL_BOUNDS = {
    'linear_regression': [
        (0, 2),        # fit_intercept (0=False, 1=True) - mapped to int
    ],
    'logistic_regression': [
        (-2.0, 2.0),   # C (Regularization): 10^-4 to 10^4 (Log Scale)
        (0, 2),        # penalty: 0=l2, 1=none (assuming lbfgs solver)
        (-5.0, -1.0),  # tol: 10^-5 to 10^-1
        (0, 3),        # 3: solver: 0=lbfgs, 1=liblinear, 2=saga
    ],
    'neural_network': [
        (10, 300),     # hidden_layer_size (neurons)
        (0, 3),        # activation: 0=relu, 1=tanh, 2=logistic
        (0, 3),        # solver: 0=adam, 1=sgd, 2=lbfgs
        (-5.0, -1.0),  # alpha: 10^-5 to 10^-1 (Log Scale)
        (-4.0, -1.0),  # learning_rate_init: 10^-4 to 10^-1 (Log Scale)
    ],
    'decision_tree': [
        (2, 100),      # max_depth
        (2, 40),       # min_samples_split
        (1, 20),       # min_samples_leaf
        (0, 2),        # criterion: 0=gini, 1=entropy
        (0, 2),        # splitter: 0=best, 1=random
    ],
    'random_forest': [
        (10, 300),     # n_estimators
        (2, 100),      # max_depth
        (2, 40),       # min_samples_split
        (1, 20),       # min_samples_leaf
        (0, 2),        # criterion: 0=gini, 1=entropy
        (0, 3),        # max_features: 0=sqrt, 1=log2, 2=None
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
    ],
}

def get_default_genes(model_name):
    defaults = {}
    
    if model_name == 'linear_regression':
        # Default: fit_intercept=True (1)
        defaults['genes'] = [1.0] 

    elif model_name == 'logistic_regression':
        # C=1.0, penalty='l2', tol=1e-4, solver='lbfgs'
        # C (10^0 = 1), penalty (0=l2), tol (10^-4), solver (0=lbfgs)
        defaults['genes'] = [0.0, 0.0, -4.0, 0.0]

    elif model_name == 'neural_network':
        # hidden=(100,), activation='relu', solver='adam', alpha=0.0001, lr_init=0.001
        defaults['genes'] = [100.0, 0.0, 0.0, -4.0, -3.0]

    elif model_name == 'decision_tree':
        # max_depth=None, split=2, leaf=1, gini, best
        # Note: GA Bound for depth is (2,100). We use 100.0 to approximate "None" (max depth).
        defaults['genes'] = [100.0, 2.0, 1.0, 0.0, 0.0]

    elif model_name == 'random_forest':
        # n_estimators=100, max_depth=None, split=2, leaf=1, gini, max_features='sqrt'
        defaults['genes'] = [100.0, 100.0, 2.0, 1.0, 0.0, 0.0]

    elif model_name == 'knn':
        # n_neighbors=5, weights='uniform', p=2
        defaults['genes'] = [5.0, 0.0, 2.0]

    elif model_name == 'svm':
        # C=1.0, kernel='linear', gamma='scale'
        # C(10^0), Kernel(0=linear), Gamma(Ignored for linear, but setting -2.0 placeholder)
        defaults['genes'] = [0.0, 0.0, -2.0] 

    return defaults.get('genes', [])

def decode_params(model_name, genes):
    params = {}
    
    if model_name == 'linear_regression':
        params['fit_intercept'] = bool(int(genes[0]))

    elif model_name == 'logistic_regression':
        params['C'] = float(10 ** genes[0])
        params['tol'] = float(10 ** genes[2])
        solver_map = {0: 'lbfgs', 1: 'liblinear', 2: 'saga'}
        solver = solver_map.get(int(genes[3]), 'lbfgs')
        params['solver'] = solver
        
        raw_penalty = int(genes[1]) 
        
        if solver == 'lbfgs':
            params['penalty'] = 'l2' if raw_penalty == 0 else 'none'
        elif solver == 'liblinear':
            params['penalty'] = 'l2' if raw_penalty == 0 else 'l1'
        elif solver == 'saga':
            params['penalty'] = 'l2' if raw_penalty == 0 else 'l1'

    elif model_name == 'neural_network':
        params['hidden_layer_sizes'] = (int(genes[0]),)
        
        act_map = {0: 'relu', 1: 'tanh', 2: 'logistic'}
        params['activation'] = act_map.get(int(genes[1]), 'relu')
        
        solver_map = {0: 'adam', 1: 'sgd', 2: 'lbfgs'}
        params['solver'] = solver_map.get(int(genes[2]), 'adam')
        
        params['alpha'] = float(10 ** genes[3])
        params['learning_rate_init'] = float(10 ** genes[4])

    elif model_name == 'decision_tree':
        params['max_depth'] = int(genes[0])
        params['min_samples_split'] = int(genes[1])
        params['min_samples_leaf'] = int(genes[2])
        
        crit_map = {0: 'gini', 1: 'entropy'}
        params['criterion'] = crit_map.get(int(genes[3]), 'gini')
        
        split_map = {0: 'best', 1: 'random'}
        params['splitter'] = split_map.get(int(genes[4]), 'best')

    elif model_name == 'random_forest':
        params['n_estimators'] = int(genes[0])
        params['max_depth'] = int(genes[1])
        params['min_samples_split'] = int(genes[2])
        params['min_samples_leaf'] = int(genes[3])
        
        crit_map = {0: 'gini', 1: 'entropy'}
        params['criterion'] = crit_map.get(int(genes[4]), 'gini')
        
        feat_map = {0: 'sqrt', 1: 'log2', 2: None}
        params['max_features'] = feat_map.get(int(genes[5]), 'sqrt')

    elif model_name == 'knn':
        params['n_neighbors'] = int(genes[0])
        
        w_map = {0: 'uniform', 1: 'distance'}
        params['weights'] = w_map.get(int(genes[1]), 'uniform')
        
        params['p'] = int(genes[2]) # 1 or 2

    elif model_name == 'svm':
        params['C'] = float(10 ** genes[0])
        
        k_map = {0: 'linear', 1: 'rbf', 2: 'poly', 3: 'sigmoid'}
        kernel = k_map.get(int(genes[1]), 'linear')
        params['kernel'] = kernel
        
        if kernel in ['rbf', 'poly', 'sigmoid']:
            params['gamma'] = float(10 ** genes[2])
        else:
            params['gamma'] = 'scale'

    return params

def ems(X, y, models, target_metric='accuracy', report=False):
    options_validation(models)
    
    best_global_model = None
    best_global_score = -float('inf')
    best_global_info = {}

    if report:
        reports_dir = "files/ga_logs"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)

        report_file = os.path.join(reports_dir, f"evolution_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        with open(report_file, "w") as f:
            f.write("Model,Generation,Score,Parameters\n")

    for model_name in models:
        print(f"Training {model_name}...")
        
        train_func = function_registry.get(model_name)
        bounds = MODEL_BOUNDS.get(model_name, [])

        print(f"   -> Evaluating default parameters for {model_name}...")
        try:
            default_result = train_func(X, y)
            if default_result and 'metrics' in default_result:
                default_score = default_result['metrics'].get(target_metric, -float('inf'))
                print(f"   -> Default Score ({target_metric}): {default_score:.4f}")
                
                if default_score > best_global_score:
                    best_global_score = default_score
                    best_global_model = default_result['model']
                    best_global_info = {
                        'model_name': model_name,
                        'score': default_score,
                        'params': 'default'
                    }
            else:
                print("   -> Default training failed or returned no metrics.")
                default_score = -float('inf')
        except Exception as e:
            print(f"   -> Error training defaults: {e}")
            default_score = -float('inf')

        if bounds:
            def fitness_function(individual):
                params = decode_params(model_name, individual)
                
                try:
                    if model_name in ['random_forest', 'knn', 'linear_regression', 'logistic_regression']:
                         params['n_jobs'] = -1
                    
                    result = train_func(X, y, **params)
                    if result and 'metrics' in result:
                        return result['metrics'].get(target_metric, -float('inf'))
                except Exception as e:
                    pass
                
                return -float('inf')

            def generation_report(gen, score, genes):
                readable_params = decode_params(model_name, genes)
                param_str = str(readable_params).replace(",", ";") 
                
                if report:
                    with open(report_file, "a") as f:
                        f.write(f"{model_name},{gen},{score:.4f},{param_str}\n")

            seed_genes = get_default_genes(model_name)
            seeds = [seed_genes] if seed_genes else None

            ga_result = run_genetic_algorithm(
                fitness_function=fitness_function,
                gene_bounds=bounds,
                seeds=seeds,
                population_size=30,
                generations=15,
                mutation_rate=0.2,
                elitism_count=3,
                maximize=True,         
                verbose=True,
                generation_report=generation_report,
            )

            if ga_result['best_score'] > best_global_score:
                best_global_score = ga_result['best_score']
                best_params = decode_params(model_name, ga_result['best_solution'])
                
                final_run = train_func(X, y, **best_params)
                best_global_model = final_run['model']
                
                best_global_info = {
                    'model_name': model_name,
                    'score': best_global_score,
                    'params': best_params
                }

    return {
        "model": best_global_model,
        "info": best_global_info
    }


def options_validation(models):
    if not all(model in function_registry for model in models):
        raise ValueError
    
