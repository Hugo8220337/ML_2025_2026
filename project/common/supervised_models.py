import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from .metrics import evaluate_model, get_classification_metrics
from xgboost import XGBClassifier

def train_linear_regression(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    random_state=42,
    fit_intercept=True,
    copy_X=True,
    n_jobs=None,
    positive=False,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, y_train = X, y

    model = LinearRegression(
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        n_jobs=n_jobs,
        positive=positive
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)
    metrics = evaluation['metrics']

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": model.coef_,
        "intercept": model.intercept_
    }

def train_logistic_regression(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    penalty='l2',
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=42,
    solver='lbfgs',
    max_iter=10000,
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y
    
    model = LogisticRegression(
        penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state,
        solver=solver, max_iter=max_iter, verbose=verbose, warm_start=warm_start,
        n_jobs=n_jobs, l1_ratio=l1_ratio
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training (check your parameters/solver compatibility): {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions},
        "intercept": model.intercept_,
        "coefficients": model.coef_
    }

def train_neural_network(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=None,
    tol=1e-4,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10,
    max_fun=15000,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
        batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
        power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol,
        verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1,
        beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions}
    }

def train_decision_tree(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y

    model = DecisionTreeClassifier(
        criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions}
    }

def train_random_forest(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y

    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
        warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions}
    }

def train_knn(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
        p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs
    )
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions}
    }

def train_svm(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    C=1.0,
    kernel='linear',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape='ovr',
    break_ties=False,
    random_state=None,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y

    model = SVC(
        C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
        probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
        verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
        break_ties=break_ties, random_state=random_state
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "intercept": model.intercept_ if hasattr(model, 'intercept_') else "N/A",
        "coefficients": model.coef_ if kernel == 'linear' else "N/A",
        "test_data": {"y_test": y_test, "predictions": predictions}
    }

def train_naive_bayes(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    distribution='gaussian',
    priors=None,
    var_smoothing=1e-9,
    alpha=1.0,
    fit_prior=True,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_random_state)
    else:
        X_train, y_train = X, y

    if distribution == 'gaussian':
        model = GaussianNB(priors=priors, var_smoothing=var_smoothing)
    elif distribution == 'multinomial':
        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=priors)
    else:
        print(f"Error: distribution '{distribution}' not supported. Choose 'gaussian' or 'multinomial'.")
        return None

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    predictions = model.predict(X_test)

    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions}
    }

def train_isolation_forest(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    random_state=42,
    n_estimators=100,
    contamination='auto',
    max_features=1.0,
    bootstrap=False,
    n_jobs=None,
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, y_train = X, y


    if hasattr(X_train, 'toarray'):
        real_indices = np.where(y_train == 0)[0]
        X_train_clean = X_train[real_indices]
    else:
        X_train_clean = X_train[y_train == 0]

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    model.fit(X_train_clean)

    metrics = {}
    if X_test is not None and y_test is not None:
        raw_preds = model.predict(X_test)
        
        preds_mapped = np.where(raw_preds == 1, 0, 1)
        
        metrics = get_classification_metrics(y_test, preds_mapped)

    return {
        "model": model,
        "metrics": metrics,
        "test_data": {"y_test": y_test, "predictions": preds_mapped if 'preds_mapped' in locals() else None}
    }

def train_xgboost(
    X,
    y,
    X_test=None,
    y_test=None,
    test_size=0.2,
    split_random_state=42,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0,
    n_jobs=-1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    **kwargs
):
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_random_state
        )
    else:
        X_train, y_train = X, y

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        n_jobs=n_jobs,
        random_state=random_state,
        use_label_encoder=use_label_encoder,
        eval_metric=eval_metric,
        **kwargs
    )

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during XGBoost training: {e}")
        return None

    predictions = model.predict(X_test)
    evaluation = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": evaluation['metrics'],
        "test_data": {"y_test": y_test, "predictions": predictions},
        "feature_importances": model.feature_importances_
    }