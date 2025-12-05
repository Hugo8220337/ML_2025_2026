import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_regression_actual_vs_predicted(y_test, predictions, title="Actual vs Predicted"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='blue')
    
    
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


def plot_feature_importances(feature_importances, title="Feature Importances"):
    if not feature_importances:
        print("No feature importances to plot.")
        return

    
    df_imp = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()



def plot_kmeans_clusters(X, labels, centroids, title="K-Means Clusters (First 2 Features)"):
    if isinstance(X, pd.DataFrame):
        data = X.values
    else:
        data = X

    plt.figure(figsize=(10, 8))
    
    
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    plt.title(title)
    plt.xlabel(f"Feature 0")
    plt.ylabel(f"Feature 1")
    plt.legend()
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.show()


def plot_pca_2d(df_transformed, target=None, title="PCA 2D Projection"):
    plt.figure(figsize=(10, 8))
    
    x_col = df_transformed.columns[0] 
    y_col = df_transformed.columns[1] 
    
    if target is not None:
        sns.scatterplot(x=x_col, y=y_col, data=df_transformed, hue=target, palette='viridis', s=60)
        plt.legend(title='Class')
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df_transformed, s=60)
        
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()



def plot_training_history(history, metric='loss', title="Training History"):
    plt.figure(figsize=(10, 6))
    
    
    if metric in history:
        plt.plot(history[metric], label=f'Train {metric.capitalize()}')
    
    
    val_metric = f'val_{metric}'
    if val_metric in history:
        plt.plot(history[val_metric], label=f'Validation {metric.capitalize()}')
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()