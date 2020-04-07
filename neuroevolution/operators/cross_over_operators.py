import numpy as np

def single_point_crossover(w1: np.array,w2: np.array):
  """Crossover operator that uses a single point as the divisor for the crossover
  operation. It makes an horizontal split between the 2 arrays provided to the
  method
  
  Arguments:
      w1 {np.array} -- 1st array of weights what is going to be used.
      w2 {np.array} -- 2nd array of weights what is going to be used.
  
  Raises:
      AttributeError: If the shapes do not match
  
  Returns:
      w1,w1 -- new arrays of weights ensambling new behaviour
  """
  if w1.shape != w2.shape:
    raise AttributeError("Shapes must be the same if we are going to crossover")
  crossover_point = np.random.randint(0, w1.shape[0])
  w1_splitted = np.split(w1, [crossover_point])
  w2_splitted = np.split(w2, [crossover_point])
  w1_ensembled = np.concatenate([w1_splitted[0], w2_splitted[1]])
  w2_ensembled = np.concatenate([w2_splitted[0], w1_splitted[1]])
  return (w1_ensembled, w2_ensembled)


