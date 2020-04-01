import numpy as np
class Neuron:
  
  """Constructor of the neuron, it receives a weight, a random initiaizer
  flag, and the size of the layer where is going to be placed to be used on the
  random intialization of the neuron.
  
  Returns:
      void
  """
  def __init__(self, size_l, weight = 0, random = False):
    if not random:
      self.weight = weight
    else:
      if size_l == 1:
        size_l=2
      [self.weight] = np.random.rand(1)*np.sqrt(2/(size_l-1))


  def calculate_output(self, input_value):
    return input_value * self.weight
