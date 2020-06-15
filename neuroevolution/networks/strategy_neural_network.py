from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.activation_functions import sigmoid
import pandas as pd
import numpy as np
import typing

class StrategyNeuralNetwork(GeneticNeuralNetwork):
  """Module that ensembles an evolutionary-strategy based neural network.
  It inherits from genetic neural network since behaviour is similar in a lot of
  ways.
  """
  def __init__(self, layers:list, num_of_classes:int, input_size:int,
               activation_functs = None, pop_size = 50, sigma = 0.5, lr = 0.1,
               verbose:bool =True):       
    self.learning_rate = lr
    self.statistics = pd.DataFrame(columns=['epoch', 'fitness'])
    self.sigma = sigma
    self.pop_size = pop_size
    self.layers = layers
    self.population = {}
    self.additions = {}
    self.verbose = verbose
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    if pop_size <1:
      raise AttributeError(
        "Can't create a genetic neural net with less than 1 individual")
    if self.learning_rate <= 0:
      raise AttributeError("The learning rate parameter can't be 0 or lower")
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases(pop_size=1)
  
  def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
    """Training method. It loops epochs times and tries to find the global 
    minima, in this case by aplying a natural evolutionary strategy. If verbose
    is present it prints every 10 epochs the loss value that contains the 
    original_solution. This solution is mutated every epoch in search for the
    global minimum in every problem.

    Arguments:
        inputs {np.ndarray} -- Inputs array
        targets {np.ndarray} -- Real values array
        epochs {int} -- Number of iterations

    Raises:
        AttributeError: If the array comes initialized with more than 1 solution
    """
    if len(self.population.keys()) != 1:
      raise AttributeError(
        "Can't have more than one solution in this state of the program")
    activated_results = self.evolved_feed_forward(inputs)
    self.calculate_loss(activated_results,targets)
    original_solution = self.population[list(self.population.keys())[0]]
    print("Initial loss: {}".format(original_solution['loss']))
    for i in range(epochs):
      if self.verbose and i % 10 ==0:
        temp_pop = self.population
        self.population = {"P0":original_solution}
        activated_results = self.evolved_feed_forward(inputs)
        self.calculate_loss(activated_results,targets)
        original_solution = self.population[list(self.population.keys())[0]]
        self.population = temp_pop
        print("Epoch {}: Original Solution Loss: {}".format(
            i, original_solution['loss']))
      self.initialize_weithts_and_biases(pop_size=self.pop_size)
      self.jitter_population(original_solution)
      activated_results = self.evolved_feed_forward(inputs)
      self.calculate_loss(activated_results,targets)
      standard_losses = self.standarize_loss()
      self.update_weights(standard_losses,original_solution)
      self.additions = {}
      self.__extract_statistics(i)
    self.population = {"final_solution": original_solution}
    activated_results = self.evolved_feed_forward(inputs)
    self.calculate_loss(activated_results,targets)
    return (self.population['final_solution']['loss'], self.statistics)

  def __extract_statistics(self, epoch: int):
    result = {'epoch': int(epoch),
        'fitness': self.population[list(self.population.keys())[0]]['loss']}
    self.statistics = self.statistics.append(result, ignore_index=True)

  def test(self, inputs: np.ndarray, labels: np.ndarray):
      """Function used to test the final resolution of the neural network

      Arguments:
          inputs {np.ndarray} -- Inputs for the algorithm
          labels {np.ndarray} -- Labels for the inputs

      Returns:
          loss -- loss of the data passed into it
      """
      activated_results = self.evolved_feed_forward(inputs)
      self.calculate_loss(activated_results,labels)
      return self.population['final_solution']['loss']

  def update_weights(self, standard_losses:np.array, original_solution:dict):
    """Method that updates the weights and biases in the original solution

    Arguments:
        standard_losses {np.array} -- List of standard losses
        original_solution {dict} -- Original solution where the weights 
        are going to be updated
    """
    for i,v in enumerate(self.population.values()):
      for (a,b) in v.items():
        original_solution[a] = original_solution[a] - self.learning_rate/(
            self.pop_size*self.sigma) * np.dot(b,standard_losses[i])
    

  def standarize_loss(self):
    """Method that normalizes the loss between 0 and 1

    Returns:
        list -- List of losses normalized
    """
    vs = list(self.population.values())
    losses = []
    for i in vs:
      losses.append(i['loss'])
    losses = np.asarray(losses)
    return (losses - np.mean(losses)) / np.std(losses)
    

  def jitter_population(self, original_solution:dict):
    """Method that generates pop_size jitters of the original solution using 
    random normal distributed noise (gaussian noise) ponderated by sigma.

    Arguments:
        original_solution {dict} -- Used to generate the jittered population
    """
    for (_,v) in self.population.items():
      jittered_individual = {}
      for (a,b) in v.items():
        jittered_individual[a] = original_solution[a] - (self.sigma * b)
      self.new_individual(jittered_individual)
    self.population = self.additions
