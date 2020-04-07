import numpy as np

def generate_rand_value(weights,low = -2, high = 2):
  index = []
  for i in len(weights.shape):
    index.append(np.random.randint(0,weights.shape[i] ))
  index =  tuple(index)
  random_value = np.random.uniform(low=low,high=high)
  return index,random_value

def randomize_a_weight(weights:np.array):
  index, random_value = generate_rand_value(weights)
  weights.put(index,random_value)
  return weights

def add_a_weighted_random_value(weights:np.array, percentage = 0.1):
  index, random_value = generate_rand_value(weights)
  weights.put(index, random_value*percentage)
  return weights

def add_or_substract_a_random_value(weights:np.array):
  index,random_value = generate_rand_value(weights,0,1)
  add_or_sbstract = np.random.randint(0,2)
  if add_or_sbstract == 0:
    add_or_sbstract = -1
  weights.put(index, (weights[index])*(random_value*add_or_sbstract))
  return weights

def change_sign_of_a_weight(weights:np.array):
  index, random_value = generate_rand_value(weights)
  weights.put(index, -weights[index])
  

