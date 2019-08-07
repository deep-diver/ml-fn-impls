import numpy as np

def cross_entropy_error(preds, labels):
  delta = 1e-7
  return -np.sum(labels*np.log(preds + delta))
