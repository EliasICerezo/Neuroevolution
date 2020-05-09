import neuroevolution.operators.mutation as mutation_operators
import neuroevolution.operators.crossover as crossover_operators
import neuroevolution.operators.selection as selection_operators
import numpy as np 
import copy

MUTATION_PROBABILITY = 40
CROSSOVER_PROBABILITY = 20


class GeneticAlgorithm():
  def __init__(self, pop_size = 100, length = 500,
    verbose = True):
    self.population = {}
    self.pop_size = pop_size
    self.length = length
    self.additions = {}
    self.verbose = verbose
    if length < 0:
      raise AttributeError(
        "Length of the string for the ONEMAX problem can't be below 0")
    if pop_size < 0:
      raise AttributeError("Population size can't be blow 0")
    self.initialize_genetic_algorithm()
      

  def initialize_genetic_algorithm(self):
    """Function that initializes the actual weights and biases of the neural net
    using the previous attributes.
    """
    np.random.seed = 42
    for pop_index in range(self.pop_size):
      population_encoded_string = "P{}".format(pop_index)
      self.population[population_encoded_string] = {}
      self.population[population_encoded_string]['array'] = list(np.random.randint(
          2, size=self.length))

  def evaluate_individuals(self):
    for (k,v) in self.population.items():
      self.population[k]['value'] = sum(v['array'])

  def sort_individuals(self):
    sorted_pop = sorted(
          self.population.items(), key=lambda x: x[1]['value'], reverse=True)
    npop = {}
    for (k,v) in sorted_pop:
      npop[k] = v
    self.population = npop
  
  def solve(self, max_iter):
    for i in range(max_iter):
      self.evaluate_individuals()
      # breakpoint()
      self.sort_individuals()
      self.population = selection_operators.selection(
          self.population,self.pop_size, division=1)
      for (k,v) in self.population.items():
        n = np.random.randint(0,100)
        if n < MUTATION_PROBABILITY:
          temp = {'array': flip_random_value(v['array'])}
          self.__new_individual(temp)
        n = np.random.randint(0,100)
        if n < CROSSOVER_PROBABILITY:
          v2 = self.population[
                (list(self.population.keys()))[
                np.random.randint(len(self.population.keys()))]]
          array_value = crossover_operators.single_point_crossover(
              np.asarray(v['array']), np.asarray(v2['array']))
          self.__new_individual({'array': array_value[0].tolist()})
          self.__new_individual({'array': array_value[1].tolist()})
      self.population.update(self.additions)
      self.additions = {}
      if self.verbose and i%50 == 0:
        print("EPOCHS: {}".format(i))
    self.evaluate_individuals()
    self.sort_individuals()
    self.population = selection_operators.selection(
          self.population,self.pop_size, division=1)
        
  
  def __new_individual(self, individual:dict):
    """Function that will add a new individual to the population
    
    Arguments:
        individual {dict} -- Object that is goig to be added to the population
    """
    self.additions[hash(str(individual))] = individual
    pass

def flip_random_value(arr:list):
  arr_copy = copy.deepcopy(arr)
  idx = np.random.randint(0,len(arr_copy))
  if arr_copy[idx] == 1:
    arr_copy[idx] = 0
  else:
    arr_copy[idx] = 1
  return arr_copy
