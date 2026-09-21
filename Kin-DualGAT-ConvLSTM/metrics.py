import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pingouin as pg
import pandas as pd

def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def compute_plcc(y_true, y_pred):
    y_t = np.array(y_true).squeeze()
    y_p = np.array(y_pred).squeeze()
    return pearsonr(y_t, y_p)[0]

def compute_srcc(y_true, y_pred):
    y_t = np.array(y_true).squeeze()
    y_p = np.array(y_pred).squeeze()
    return spearmanr(y_t, y_p)[0]

def compute_icc(data_frame):
    # Computes two-way random effects model, absolute agreement
    icc = pg.intraclass_corr(data=data_frame, targets='target', raters='rater', ratings='rating')
    return icc.set_index('Type').loc['ICC2']['ICC']

def participant_clustered_bootstrap(y_true, y_pred, participants, n_iterations=5000):
    # Bootstrapping logic here
    pass