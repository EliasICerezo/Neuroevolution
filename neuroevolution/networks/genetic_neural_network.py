from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
from neuroevolution.activation_functions import sigmoid
from neuroevolution.error_functions import crossentropy_loss
from neuroevolution.operators.crossover import single_point_crossover 
import neuroevolution.operators.selection as selection_operators
import neuroevolution.operators.mutation as mutations
import numpy as np
import inspect
import random
import copy

# Genetic operator probabilities in percentage
SELECTION_PROBABILITY = 25
MUTATION_PROBABILITY = 40
CROSSOVER_PROBABILITY = 20


class GeneticNeuralNetwork(BasicNeuralNetwork):
  """This class ensembles the behaviour of a neuroevolved neural network.
  It is evolved via a genetic algorithm
  """
  def __init__(self, layers:list, num_of_classes:int, input_size:int,
               activation_functs = None, pop_size = 10, max_pop_size = 100, 
               verbose = True):
    """Constructor of the Genetic Neural Network
    
    Arguments:
        layers {list} -- A list that ensembles the number of nodes of each layer
        of the neural network
        num_of_classes {int} -- The number of classes that is going to output 
        the neural network
        input_size {int} -- Size of the input layer
    
    Keyword Arguments:
        activation_functs {[type]} -- A list of the activation functions used
        by each layer of the Genetic Neural Network (default: {None})
        pop_size {int} -- Size of the initial population (default: {10})
        max_pop_size {int} -- Maximum number of elements in the population
         (default: {100})
    
    Raises:
        AttributeError: If any of the given arguments does not agree with the 
        conventions
    """
    self.verbose = verbose
    self.layers = layers
    self.population = {}
    self.pop_size = pop_size
    self.max_pop_size = max_pop_size
    self.additions = {}

    if pop_size == 0:
      raise AttributeError(
        "Can't create a genetic neural net without a single individual")
    if max_pop_size == 0:
      raise AttributeError(
        "Can't create a genetic neural net without a 0 maximum individuals")
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    if pop_size > max_pop_size:
      raise AttributeError(
        "The size of the population can't be greater than its maximum")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases()
  
  def initialize_weithts_and_biases(self, pop_size = 0):
    """Function that initializes the actual weights and biases of the neural net
    using the previous attributes.
    """
    if pop_size == 0:
      pop_size = self.pop_size
    np.random.seed = 42
    for pop_index in range(pop_size):
      population_encoded_string = "P{}".format(pop_index)
      self.population[population_encoded_string] = {}
      for i, e in enumerate(self.layers):
        if i < len(self.layers)-1:
          self.population[population_encoded_string][
                          'W{}'.format(i+1)] = np.random.randn(self.layers[i],
                          self.layers[i+1])
          self.population[population_encoded_string][
                          'b{}'.format(i+1)] = np.random.randn(self.layers[i+1])

  def evolved_feed_forward(self, inputs: np.array):
    """Calculation of the forward pass of the neural network with the clear 
    distinction that it needs to be calculated for each of the population 
    genotypes
    
    Arguments:
        inputs {np.array} -- The input NP Array that ensembles the dataset
    
    Returns:
        population_activated_results -- The activated results calculation
    """
    population_activated_results = {}
    for k in self.population.keys():
      y_hat = self.calculate_feed_forward(inputs, self.population[k])
      population_activated_results[k] = y_hat
    return population_activated_results
  
  def train(self, inputs: np.ndarray, targets: np.ndarray, max_iter: int):
    """Training function that defines the evolutive behaviour of the neural net.
    
    Arguments:
        inputs {np.ndarray} -- Input arrays in the form of arrays
        targets {np.ndarray} -- The values desired fot those inputs
        max_iter {int} -- Maximum number of iterations that the nnet will be 
        running.
    """
    for i in range(max_iter):
      activated_results = self.evolved_feed_forward(inputs)
      self.calculate_loss(activated_results, targets)
      
      # Selection operator
      if (
          len(self.population.keys()) >
          self.max_pop_size or np.random.randint(0,101) <
          SELECTION_PROBABILITY):
        self.population = selection_operators.selection(
            self.population, self.max_pop_size)

      # Mutation operator (it mutates weights and biases)
      self.mutate_population()
      # Crossover operator
      self.crossover_populations('W')
      self.crossover_populations('b')
      self.population.update(self.additions)
      self.additions = {}
      self.sort_population()
      if self.verbose:
        if i % 50 == 0:
          print("Epochs: {}".format(i))
    return self.population[list(self.population.keys())[0]]['loss']

  def test(self, inputs: np.ndarray, labels: np.ndarray):
    """Function used to test the final resolution of the neural network

    Arguments:
        inputs {np.ndarray} -- Inputs for the algorithm
        labels {np.ndarray} -- Labels for the inputs

    Returns:
      loss -- loss of the data passed into it
    """
    temp = self.population
    self.population = {'test_solution' : self.population[list(self.population.keys())[0]]}

    activated_results = self.evolved_feed_forward(inputs)
    self.calculate_loss(activated_results, labels)
    loss = self.population[list(self.population.keys())[0]]['loss']
    self.population = temp
    return loss 

  def calculate_loss(self, activated_results: np.ndarray, targets: np.ndarray):
    """This function calculates the loss for the population of the network
    
    Arguments:
        activated_results {numpy.ndarray} -- Array resulting of the forward pass
        targets {np.ndarray} -- Array of true values passed during the training 
        process
    """
    for k,v in activated_results.items():
      loss = crossentropy_loss(targets,v)
      self.population[k]['loss']=loss

  def mutate_population(self, sigma: float = None):
    """Function that performs a random mutarion in the population, the mutated 
    structures can be either the weights, the biases, or both. It obeys to a 
    mutation probability that is constant during the training process
    
    Keyword Arguments:
        sigma {float} -- If present, applies certain mutation ponderated rather
        than a random mutaion operattion (default: {None})
    """
    for v in self.population.values():
      for i in range(len(self.layers)-1):
        if np.random.randint(0,100) < MUTATION_PROBABILITY:
          v_copy = copy.deepcopy(v)
          elem = v_copy['W{}'.format(i+1)]
          elem = self.__mutation_operator(elem,sigma)
          v_copy['W{}'.format(i+1)] = elem
          self.new_individual(v_copy)
        if np.random.randint(0,100) < MUTATION_PROBABILITY:
          v_copy = copy.deepcopy(v)
          elem = v_copy['b{}'.format(i+1)]
          elem = self.__mutation_operator(elem,sigma)
          v_copy['b{}'.format(i+1)] = elem
          self.new_individual(v_copy)

  def sort_population(self):
    """Function that sorts the population over loss values ascending
    """
    sorted_pop = sorted(
          self.population.items(), key=lambda x: x[1]['loss'], reverse=False)
    npop = {}
    for (k,v) in sorted_pop:
      npop[k] = v
    self.population = npop

  def new_individual(self, individual:dict):
    """Function that will add a new individual to the population
    
    Arguments:
        individual {dict} -- Object that is goig to be added to the population
    """
    self.additions[hash(str(individual))] = individual
    pass

  def crossover_populations(self, key):
    """Crossover operator. At the moment the only one available is the single
    point one, but they will come more
    
    Arguments:
        key {string} -- The key to crossover, rather it can be W for weights or
        b for biases.
    """
    [p1,p2] = random.sample(self.population.keys(), k=2)
    for i in range(len(self.layers) - 1):
      if np.random.randint(0,100) < CROSSOVER_PROBABILITY:
        p1_copy = copy.deepcopy(self.population[p1])
        p2_copy = copy.deepcopy(self.population[p2])
        w_1 = self.population[p1]['{}{}'.format(key,i+1)]
        w_2 = self.population[p2]['{}{}'.format(key,i+1)]
        new_elements = single_point_crossover(w_1, w_2)
        p1_copy['{}{}'.format(key,i+1)] = new_elements[0]
        p2_copy['{}{}'.format(key,i+1)] = new_elements[1]
        self.new_individual(p1_copy)
        self.new_individual(p2_copy)

  
  def __mutation_operator(self, elem:np.array, sigma: float = None):
    """A function that applies the mutation operator to the given element wether
    it be a bias or a weight
    
    Arguments:
        elem {np.array} -- The element to be mutated
        sigma {float} -- (optional) Value used to ponderate the mutation
    
    Returns:
        elem -- The mutated element
    """
    if sigma is None:
      m_operators = inspect.getmembers(mutations, inspect.isfunction)
      operation = random.choice(m_operators)
      while operation[0] == 'generate_rand_value':
        m_operators = inspect.getmembers(mutations, inspect.isfunction)
        operation = random.choice(m_operators)
      # At this moment, operation is a tuple where the second elem is the 
      # operation itself
      operation = operation[1]
      res = operation(elem)
    else:
      res = mutations.add_a_weighted_random_value(elem,sigma)
    return res
