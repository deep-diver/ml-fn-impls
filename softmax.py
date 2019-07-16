import numpy as np

def softmax(one_demension_array):
  exps = np.exp(one_demension_array)
  exps_sum = np.sum(exps)
  
  y = exps / exps_sum
  return y
