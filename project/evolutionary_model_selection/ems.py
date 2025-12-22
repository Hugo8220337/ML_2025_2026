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
    'linear_regression': [],
    'logistic_regression': [
        (0.01, 10.0), # C (float, regularization strength)
    ],
    'neural_network': [
        (10, 200),    # hidden_layer_size 
        (0.0001, 0.1) # learning_rate_init 
    ],
    'decision_tree': [
        (2, 50),      # max_depth 
        (2, 20),      # min_samples_split 
        (1, 10),      # min_samples_leaf 
    ],
    'random_forest': [
        (10, 200),    # n_estimators 
        (2, 30),      # max_depth 
        (2, 20),      # min_samples_split 
    ],
    'knn': [
        (1, 30),      # n_neighbors 
    ],
    'svm': [
        (0.1, 100.0), # C 
    ],
}


def decode_params(model_name, genes):
    params = {}
    
    if model_name == 'random_forest':
        params['n_estimators'] = int(genes[0])
        params['max_depth'] = int(genes[1])
        params['min_samples_split'] = int(genes[2])
        
    elif model_name == 'decision_tree':
        params['max_depth'] = int(genes[0])
        params['min_samples_split'] = int(genes[1])
        params['min_samples_leaf'] = int(genes[2])
        
    elif model_name == 'svm':
        params['C'] = genes[0]

        
    elif model_name == 'knn':
        params['n_neighbors'] = int(genes[0])
        
    elif model_name == 'logistic_regression':
        params['C'] = genes[0]
        
    elif model_name == 'neural_network':
        params['hidden_layer_sizes'] = (int(genes[0]),)
        params['learning_rate_init'] = genes[1]

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

        if not bounds:
            result = train_func(X, y)
            if result:
                score = result['metrics'].get(target_metric, -float('inf'))
                
                if score > best_global_score:
                    best_global_score = score
                    best_global_model = result['model']
                    best_global_info = {
                        'model_name': model_name,
                        'score': score,
                        'params': 'default'
                    }
            continue

        def fitness_function(individual):
            params = decode_params(model_name, individual)
            
            try:
                result = train_func(X, y, **params)
                if result and 'metrics' in result:
                    return result['metrics'].get(target_metric, -float('inf'))
            except Exception as e:
                pass
            
            return -float('inf')

        def generation_report(gen, score, genes):
            readable_params = decode_params(model_name, genes)
            param_str = str(readable_params).replace(",", ";") 
            
            with open(report_file, "a") as f:
                f.write(f"{model_name},{gen},{score:.4f},{param_str}\n")

        ga_result = run_genetic_algorithm(
            fitness_function=fitness_function,
            gene_bounds=bounds,
            population_size=10,
            generations=5,
            maximize=True,         
            verbose=True,
            generation_report=generation_report
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
    
