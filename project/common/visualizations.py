import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def plot_model_comparison(results, save_path=None, title="Model Performance Comparison"):
    other = results.get('other', {})
    
    model_names = []
    scores = []
    
    for model_name, model_data in other.items():
        model_names.append(model_name.upper())
        if 'info' in model_data and 'score' in model_data['info']:
            score = model_data['info']['score']
        elif 'score' in model_data:
            score = model_data['score']
        else:
            score = 0
        scores.append(float(score))
    
    if not model_names:
        print("No model results to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if s == max(scores) else '#3498db' for s in scores]
    
    bars = plt.bar(model_names, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, max(scores) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Best Model'),
                       Patch(facecolor='#3498db', edgecolor='black', label='Other Models')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_clusters(X_reduced, labels, model_name, save_path=None, title=None):
    if X_reduced is None or labels is None:
        print("No data or labels to plot.")
        return
    
    if hasattr(X_reduced, 'values'):
        data = X_reduced.values
    elif hasattr(X_reduced, 'toarray'):
        data = X_reduced.toarray()
    else:
        data = np.array(X_reduced)
    
    if data.ndim == 1:
        print("Data must be at least 2D for plotting.")
        return
    
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, max(n_clusters, 1)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c='gray', marker='x', s=30, alpha=0.5, label='Noise')
        else:
            color_idx = label % len(colors)
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c=[colors[color_idx]], s=50, alpha=0.6, 
                       label=f'Cluster {label}')
    
    if title is None:
        title = f"{model_name.upper()} Clusters"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

def _get_test_data(model_data):
    if 'test_data' in model_data:
        return model_data['test_data']
    elif 'info' in model_data and 'test_data' in model_data['info']:
        return model_data['info']['test_data']
    return None

def _get_score(model_data):
    if 'score' in model_data:
        return model_data['score']
    elif 'info' in model_data and 'score' in model_data['info']:
        return model_data['info']['score']
    return 0.0



def plot_anomaly_confusion_matrix(results, save_path=None):
    models_data = results.get('other', {})
    if not models_data:
        print("No model results found to plot.")
        return

    n_models = len(models_data)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, models_data.items()):
        
        test_data = _get_test_data(data)

        if test_data is None:
            ax.text(0.5, 0.5, "No Test Data", ha='center')
            ax.set_title(model_name.upper())
            continue
            
        y_true = test_data.get('y_test')
        y_pred = test_data.get('predictions')

        if y_true is None or y_pred is None:
            ax.text(0.5, 0.5, "Invalid Data", ha='center')
            continue

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'],
                    cbar=False)
        
        score = _get_score(data)
        ax.set_title(f"{model_name.upper()}\nF1: {score:.4f}", fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_anomaly_scatter(X, results, save_path=None):
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
        
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)

    models_data = results.get('other', {})
    if not models_data:
        print("No model results found.")
        return

    n_plots = 1 + len(models_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1: axes = [axes]

    
    def _scatter(ax, labels, title):
        
        ax.scatter(X_pca[labels==0, 0], X_pca[labels==0, 1], 
                   c='dodgerblue', alpha=0.5, label='Real', s=15, edgecolor='w', linewidth=0.5)
        
        ax.scatter(X_pca[labels==1, 0], X_pca[labels==1, 1], 
                   c='crimson', alpha=0.6, label='Fake', s=15, edgecolor='w', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    
    y_true_ref = None
    for data in models_data.values():
        td = _get_test_data(data)
        if td and 'y_test' in td:
            y_true_ref = td['y_test']
            break
            
    if y_true_ref is not None:
        _scatter(axes[0], y_true_ref, "Ground Truth (Actual)")
    else:
        axes[0].text(0.5, 0.5, "No Ground Truth Found", ha='center')

    
    for i, (model_name, data) in enumerate(models_data.items()):
        test_data = _get_test_data(data)
        if test_data and 'predictions' in test_data:
            preds = test_data['predictions']
            _scatter(axes[i+1], preds, f"{model_name.upper()} Predictions")
        else:
            axes[i+1].text(0.5, 0.5, "No Predictions", ha='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_stance_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    labels = ['agree', 'disagree', 'discuss', 'unrelated']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(9, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
    
    plt.title(f'{model_name.upper()} - Stance Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Stance', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Stance', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()