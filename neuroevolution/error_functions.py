import numpy as np

def MSE(x, y):
  """Mean Squared Error function.
  x = nnet predictions
  y = real values of these predictions
  """
  if len(y) == len(x):
    error = [ (x[i] - y[i])**2 for i in range(0, len(x))]
    error = sum(error)
    error = error / len(y)
    return error
  else:
    raise AttributeError("Input lengths do not match")

def crossentropy_loss(y, y_hat):
    """This loss function will be used to optimize the weights on the neural net
    It uses an entropy based formula.
    
    Arguments:
        y {numpy array} -- They are the true values of the samples
        y_hat {numpy array} -- The predicted values of the feature set
    Returns 
    """
    n_sample = len(y)
    loss = -1/n_sample * (np.sum(np.multiply(np.log(y_hat), y) +
                          np.multiply((1 - y), np.log(1 - y_hat)) ))
    return loss