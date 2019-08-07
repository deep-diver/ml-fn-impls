import numpy as np

def cross_entropy_error(preds, labels):
  if preds.ndim == 1:
    preds = preds.reshape(1, preds.size)
    labels = labels.reshape(1, labels.size)
  
  batch_size = preds.shape[0]
  delta = 1e-7
  return -np.sum(np.log(preds[np.arange(batch_size), labels] + delta)) / batch_size
