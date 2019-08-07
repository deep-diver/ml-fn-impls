import numpy as np

def mean_squared_error(preds, labels):
  return 0.5 * np.sum((preds-labels)**2)
