from sklearn.metrics import mean_squared_error
import numpy as np

def get_MSE(pred, real):
    return mean_squared_error(real.flatten(), pred.flatten())

def print_metrics(pred, real):
    mse = get_MSE(pred, real)

    print('Test: MSE={:.6f}'.format(mse))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))
    
def get_MAPE(pred, real):
    ori_real = real.copy()
    epsilon = 1 # if use small number like 1e-5 resulting in very large value
    real[real == 0] = epsilon 
    return np.mean(np.abs(ori_real - pred) / real)