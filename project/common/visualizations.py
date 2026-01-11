import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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