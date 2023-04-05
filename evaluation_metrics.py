import numpy as np

"""
Implements the optimism and pessimism ratios defined in Application of LSTM, GRU 
and ICA for Stock Price Prediction by Sethia, A. and Raut, P. (ICTIS, 2018)
"""

def optimism_ratio(y_true, y_pred):
    return np.mean(y_pred > 1.015 * y_true)
  
def pessimism_ratio(y_true, y_pred):
    return np.mean(y_pred < 0.985 * y_true)
