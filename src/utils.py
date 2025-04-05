import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

def train_test_split_with_time(X, y, train_ratio=0.8, random_state=None):
    """
    Split data into train and test sets while preserving temporal order.
    
    Args:
        X: Features
        y: Labels
        train_ratio: Ratio of training data (default: 0.8)
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    
    indices = np.arange(n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )
