from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
import neuroevolution.activation_functions as activation_funtions
import neuroevolution.operators.mutation as mutation
import neuroevolution.error_functions as error_functions
import numpy as np

MUTATION_PROBABILITY = 40


class AnnealedNeuralNetwork(BasicNeuralNetwork):
  def __init__(self, layers: list, num_of_classes: int, input_size: int,
               temperature = 100, decay = 0.05, boltzmann_constant = 10, activation_functs = None):
    """Constructor of the simulated annealing-optimized neural network.
    
    Arguments:
        layers {list} -- A list of integers indicating the stricture of the 
        fully connected network to be created
        num_of_classes {int} -- Number of classes that the dataset contains
        input_size {int} -- Number of nodes in the input layer
    
    Keyword Arguments:
        temperature {float} -- Base termperature of the system (default: {100})
        decay {float} -- Decay rate of the temperature during the execution
        (default: {0.1})
        activation_functs {list} -- List of the activation functions to be used
        in the training structure (default: {None})
    """
    self.layers = layers
    if decay > 1:
      raise AttributeError("Decay can't be over 1")
    self.decay = 1 - decay
    self.params = {}
    self.loss = []
    self.boltzmann_constant = boltzmann_constant
    self.temperature = temperature
    self.init_range = None
    self.minimal_cost_configuration = {}
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [activation_funtions.sigmoid for i in range(len(self.layers))]
    self.initialize_weithts_and_biases(self.params)
  
  def train(self, inputs: np.ndarray, labels: np.ndarray, max_iter: int):
    """Function that trains (optimizes the weights and biases) of the neural net
    
    Arguments:
        inputs {np.ndarray} -- NumPy array that contains the the features to be
        used in the prediction.
        labels {np.ndarray} -- NumPy array containing the true values for the 
        given inputs.
        max_iter {int} -- Maximum number of iterations that the algorithm is
        going to be doing before stopping the execution.
    """
    activated_results = self.feed_forward(inputs)
    cost = error_functions.crossentropy_loss(labels, activated_results)
    self.minimal_cost_configuration = self.params
    self.minimal_cost_configuration['loss'] = cost
    for i in range(max_iter):
      self.temperature = self.temperature * self.decay
      # Mutate weights and biases with certain probability
      new_state = {}
      for j in range(len(self.layers)-1):
        new_state['W{}'.format(j+1)] = self.random_mutation(
            self.params['W{}'.format(j+1)])
        new_state['b{}'.format(j+1)] = self.random_mutation(
            self.params['b{}'.format(j+1)])
      new_activated_results = self.calculate_feed_forward(inputs,new_state)
      new_cost = error_functions.crossentropy_loss(labels, new_activated_results)
      if new_cost < cost:
        self.minimal_cost_configuration = new_state
        self.minimal_cost_configuration['cost'] = new_cost
      if self.accept_result(cost, new_cost, self.temperature):
        cost = new_cost
        activated_results = new_activated_results
        self.params.update(new_state)
      self.loss.append(cost)

    def test(self, inputs: np.ndarray, labels: np.ndarray):
    """Function used to test the final resolution of the neural network

    Arguments:
        inputs {np.ndarray} -- Inputs for the algorithm
        labels {np.ndarray} -- Labels for the inputs

    Returns:
        loss -- loss of the data passed into it
    """
    activated_results = self.feed_forward(inputs)
    cost = error_functions.crossentropy_loss(labels, activated_results)
    return cost

  def random_mutation(self, values:np.ndarray):
    """Function that privides a random mutation in the weights or biases.
    
    Arguments:
        values {np.ndarray} -- Weights or biases object to be mutated
    
    Returns:
        np.ndarray -- Te weights or biases array mutated
    """ 
    if np.random.randint(0,100) < MUTATION_PROBABILITY:
      values = mutation.add_a_weighted_random_value(values, np.random.uniform(0,1))
    return values

  def accept_result(self, cost, new_cost, temperature):
    """Function that calculates the acceptance probability of the provided 
    solution.
    
    Arguments:
        cost {float} -- The value of the cost function in the prevoius 
        iteration
        new_cost {float} -- The value of the cost function in the current 
        iteration
        temperature {float} -- Value of the temperature
    
    Returns:
        v -- A value between 0 and 1 repressenting the probability of being
        accepted
    """
    if new_cost <= cost:
        return True
    else:
        cost_increment = new_cost - cost
        r = np.random.rand()
        if r > np.exp(
              ((-cost_increment)/self.boltzmann_constant*self.temperature)):
                return True
        else:
          return False