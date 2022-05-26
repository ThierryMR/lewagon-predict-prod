
import numpy as np
from sklearn.metrics import r2_score, recall_score, accuracy_score

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
