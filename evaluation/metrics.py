import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics for Multi-Class Classification (3 Classes).
    Uses Macro Average to treat all classes equally.
    """
    # Ensure inputs are integers
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    unique_classes = np.unique(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred) # Shape depends on classes present
    
    return {
        'accuracy': acc,
        'precision': prec, 
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """
    Plots a multi-class confusion matrix.
    Labels: 0=Normal, 1=Anomaly, 2=Storm
    """
    plt.figure(figsize=(6, 5))
    labels = ['Normal', 'Anomaly', 'Storm']
    
    # Adjust labels based on actual CM shape if some classes are missing
    # giving a best effort annotation
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels[:cm.shape[1]], 
                yticklabels=labels[:cm.shape[0]])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_convergence(history_pso, history_bp=None, title='PSO Convergence'):
    """
    Plots the loss optimization history.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history_pso, label='Adaptive PSO', color='blue', linewidth=2)
    
    if history_bp:
        plt.plot(history_bp, label='Backpropagation (Baseline)', color='grey', linestyle='--')
        
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Fitness)')
    plt.legend()
    plt.grid(True)
    plt.show()
