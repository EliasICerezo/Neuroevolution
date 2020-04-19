import numpy as np

def generate_rand_value(weights,low = -2, high = 2):
  """A function that generates a random value and index just for using them in 
  the substitution funcions blow
  
  Arguments:
      weights {np.array} -- The weight array where the function will choose a 
      position
  
  Keyword Arguments:
      low {int} -- The lower bound to generate the random value (default: {-2})
      high {int} -- The higher bound to generate the random value (default: {2})
  
  Returns:
      index, random_value -- The selected index of the weights to be changed and
      the random number generated with the bounds above
  """
  index = []
  for i in range(len(weights.shape)):
    index.append(np.random.randint(0,weights.shape[i] ))
  index =  tuple(index)
  random_value = np.random.uniform(low=low,high=high)
  return index,random_value


# Avoid to use prepositions in method and class names. Please, you might
# consider to refactorize it from `randomize_a_weight` to `randomize_weight`
# Same for the rest of the method in the code.
def randomize_a_weight(weights:np.array):
  """Mutates the weights changing one of them for a random value between a scale
  
  Arguments:
      weights {np.array} -- The weights of the neural network to be mutated
  
  Returns:
      weights -- The weights with the mutation 
  """
  index, random_value = generate_rand_value(weights)
  weights.put(index,random_value)
  return weights

def add_a_weighted_random_value(weights:np.array, percentage = 0.1):
  """Mutates a single weight by a factor
  
  Arguments:
      weights {np.array} -- The weights array to be mutated
  
  Keyword Arguments:
      percentage {float} -- The percentage that the weights will be mutated by 
      the random value generated (default: {0.1})
  
  Returns:
      weights -- The mutated weight array.
  """
  index, random_value = generate_rand_value(weights)
  weights.put(index, random_value*percentage)
  return weights

def add_or_substract_a_random_value(weights:np.array):
  """Mutation operator that adds or substracts a random value to the weight
  
  Arguments:
      weights {np.array} -- weight array
  
  Returns:
      weights -- Updated weights
  """
  index,random_value = generate_rand_value(weights,0,1)
  add_or_sbstract = np.random.randint(0,2)
  if add_or_sbstract == 0:
    add_or_sbstract = -1
  weights.put(index, (weights[index])*(random_value*add_or_sbstract))
  return weights

def change_sign_of_a_weight(weights:np.array):
  """Changes the sign of a weight
  
  Arguments:
      weights {np.array} -- weight array
  """
  index, _ = generate_rand_value(weights)
  weights.put(index, -weights[index])
  return weights
  

