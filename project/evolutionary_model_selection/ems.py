from common.supervised_models import *
from common.unsupervised_models import *
from common.deep_learning import *


function_registry = {
    'linear_regression': train_linear_regression,
    'logistic_regression': train_logistic_regression,
    'neural_network': train_neural_network,
    'decision_tree': train_decision_tree,
    'random_forest': train_random_forest,
    'knn': train_knn,
    'svm': train_svm,
    'kmeans': train_kmeans,
    'pca': perform_pca
}

def ems(X, y, models=['linear_regression']):
    options_validation(models)
    

    for model_name in models:
        train_model = function_registry.get(model_name)
        train_model(X, y)




    
    return



def options_validation(models):
    if not all(model in function_registry for model in models):
        raise ValueError
    

