from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
from neuroevolution.activation_functions import sigmoid
from neuroevolution.operators.mutation_operators import add_a_weighted_random_value
from neuroevolution.error_functions import crossentropy_loss
import numpy as np

MUTATION_PROBABILITY = 50
WEIGHTED_MUTATION = 0.1

class AnnealedNeuralNetwork(BasicNeuralNetwork):
  def __init__(self, layers, num_of_classes, input_size, temperature = 100,
               decay = 0.1, activation_functs = None): 
    self.layers = layers
    self.decay = decay
    self.params = {}
    self.loss = []
    self.temperature = temperature
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases()
  
  def train(self, inputs: np.ndarray, labels: np.ndarray, max_iter: int):
    activated_results = self.feed_forward(inputs)
    cost = crossentropy_loss(labels, activated_results)
    for i in range(max_iter):
      self.update_temperature(i/float(max_iter))
      # Mutate weights and biases with certain probability
      new_state = {}
      for j in range(len(self.layers)-1):
        new_state['W{}'.format(j+1)] = self.random_mutation(
            self.params['W{}'.format(j+1)])
        new_state['b{}'.format(j+1)] = self.random_mutation(
            self.params['b{}'.format(j+1)])
      new_activated_results = self.calculate_feed_forward(inputs,new_state)
      new_cost = crossentropy_loss(labels, new_activated_results)
      if self.acceptance_probability(
          cost, new_cost, self.temperature) > np.random.random():
        # breakpoint()
        cost = new_cost
        activated_results = new_activated_results
        self.params.update(new_state)
      self.loss.append(cost)


  def random_mutation(self, values:np.ndarray):
    if np.random.randint(0,100) < MUTATION_PROBABILITY:
      values = add_a_weighted_random_value(values, WEIGHTED_MUTATION)
    return values
  
  def update_temperature(self, fraction):
    self.temperature = self.temperature - (self.temperature*fraction)

  def acceptance_probability(self, cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        # p = np.exp(- (new_cost - cost) / (temperature/ 100))
        # return p
        return 0
