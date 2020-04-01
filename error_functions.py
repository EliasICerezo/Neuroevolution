

"""Mean Squared Error function.
x = nnet predictions
y = real values of these predictions
"""
def MSE(x, y):
  error = [ (x[i] - y[i])**2 for i in range(0, len(x))]
  error = sum(error)
  error = error / len(y)
  return error
  


