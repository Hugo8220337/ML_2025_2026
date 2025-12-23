import os
import sys
import json
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from evolutionary_model_selection.genetic_algorithm import run_genetic_algorithm
from common.supervised_models import *
from common.unsupervised_models import *
from common.deep_learning import *
import datetime

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


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
        (-4.0, 4.0),   # C (Regularization): 10^-4 to 10^4 (Log Scale) - wider range
        (0, 2),        # penalty: 0=l2, 1=l1 (for saga), 2=none
        (-5.0, -1.0),  # tol: 10^-5 to 10^-1
        (0, 2),        # solver: 0=lbfgs, 1=saga (removed deprecated liblinear)
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
        solver_map = {0: 'lbfgs', 1: 'saga'}  
        solver = solver_map.get(int(genes[3]), 'lbfgs')
        params['solver'] = solver
        
        raw_penalty = int(genes[1])  # 0=l2, 1=l1, 2=none
        
        if solver == 'lbfgs':
            # lbfgs only supports l2 or None
            params['penalty'] = 'l2' if raw_penalty == 0 else None
        elif solver == 'saga':
            # saga supports l1, l2, elasticnet, or None
            penalty_map = {0: 'l2', 1: 'l1', 2: None}
            params['penalty'] = penalty_map.get(raw_penalty, 'l2')

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

        report_file = os.path.join(reports_dir, f"evolution_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_data = []

    for model_name in models:
        print(f"Training {model_name}...")
        
        train_func = function_registry.get(model_name)
        bounds = MODEL_BOUNDS.get(model_name, [])

        if bounds:
            failed_evaluations = [0]
            eval_counter = [0]
            
            n_samples = X.shape[0]
            max_fitness_samples = min(5000, int(n_samples * 0.1))
            
            if n_samples > max_fitness_samples:
                print(f"   -> Using {max_fitness_samples} samples for evaluation (full: {n_samples})")
                sys.stdout.flush()
                
                X_sub, _, y_sub, _ = train_test_split(
                    X, y, train_size=max_fitness_samples, random_state=42
                )
            else:
                X_sub, y_sub = X, y

            
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=42
            )
            
            print(f"   -> Evaluating default parameters for {model_name}...")
            try:
                default_result = train_func(X_train, y_train, X_test=X_test, y_test=y_test)
                if default_result and 'metrics' in default_result:
                    default_score = default_result['metrics'].get(target_metric, -float('inf'))
                    print(f"   -> Default Score ({target_metric}): {default_score:.4f}")
                else:
                    print("   -> Default training failed or returned no metrics.")
                    default_score = -float('inf')
            except Exception as e:
                print(f"   -> Error training defaults: {e}")
                default_score = -float('inf')
            
            def fitness_function(individual):
                eval_counter[0] += 1
                params = decode_params(model_name, individual)
                
                try:
                    result = train_func(X_train, y_train, X_test=X_test, y_test=y_test, **params)
                    if result and 'metrics' in result:
                        score = result['metrics'].get(target_metric, -float('inf'))
                        print(f"   [Eval {eval_counter[0]}] Score: {score:.4f}", end='\r')
                        sys.stdout.flush()
                        return score
                except Exception as e:
                    failed_evaluations[0] += 1
                    if failed_evaluations[0] <= 3:
                        print(f"   [DEBUG] Fitness eval failed for {model_name}: {e}")
                
                return -float('inf')

            def generation_report(gen, scores, population):
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                best_genes = population[best_idx]
                best_params = decode_params(model_name, best_genes)
                
                generation_data = {
                    "model": model_name,
                    "generation": gen,
                    "stats": {
                        "best_score": best_score,
                        "avg_score": float(np.mean(scores[scores > -float('inf')])),
                        "min_score": float(np.min(scores[scores > -float('inf')])) if np.any(scores > -float('inf')) else None,
                        "max_score": float(np.max(scores)),
                        "failed_evals": int(np.sum(scores == -float('inf')))
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
                population_size=15,
                generations=25,
                mutation_rate=0.3,
                crossover_rate=0.8,
                tournament_size=3,
                maximize=True,
                verbose=True,
                patience=12,
                min_delta=0.0001,
                elitism_count=1,
                mutation_strength=0.3,
                generation_report=generation_report,
            )

            print(f"\n   -> GA Best Score: {ga_result['best_score']:.4f} vs Default: {default_score:.4f}")
            
            if ga_result['best_score'] > default_score:
                best_params = decode_params(model_name, ga_result['best_solution'])
                
                final_run = train_func(X, y, **best_params)
                final_score = final_run['metrics'].get(target_metric, ga_result['best_score'])
                
                if final_score > best_global_score:
                    best_global_score = final_score
                    best_global_model = final_run['model']
                    best_global_info = {
                        'model_name': model_name,
                        'score': final_score,
                        'params': best_params
                    }
            else:
                print(f"   -> Default params are optimal. Training final model on full data...")
                final_run = train_func(X, y)
                final_score = final_run['metrics'].get(target_metric, default_score)
                
                if final_score > best_global_score:
                    best_global_score = final_score
                    best_global_model = final_run['model']
                    best_global_info = {
                        'model_name': model_name,
                        'score': final_score,
                        'params': 'default'
                    }

    if report and report_data:
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

    return {
        "model": best_global_model,
        "info": best_global_info
    }


def options_validation(models):
    if not all(model in function_registry for model in models):
        raise ValueError
    
