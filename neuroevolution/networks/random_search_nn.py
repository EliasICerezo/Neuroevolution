from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
from neuroevolution.error_functions import crossentropy_loss
import numpy as np
class RandomSearchNeuralNetwork(BasicNeuralNetwork):
  
  def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
    best_individual = self.params
    activated_results = self.calculate_feed_forward(inputs, best_individual)
    best_individual['loss'] = crossentropy_loss(targets, activated_results)
    self.loss.append(best_individual['loss'])
    for _ in range(epochs):
      new_individual = {}
      self.initialize_weithts_and_biases(new_individual)
      activated_results = self.calculate_feed_forward(inputs, new_individual)
      new_individual['loss'] = crossentropy_loss(targets,activated_results)
      if new_individual['loss'] < best_individual['loss']:
        best_individual = new_individual
        self.loss.append(new_individual['loss'])
    self.params = best_individual
    return self.params['loss']

  def test(self, inputs: np.ndarray, labels: np.ndarray):
    """Function used to test the final resolution of the neural network

    Arguments:
        inputs {np.ndarray} -- Inputs for the algorithm
        labels {np.ndarray} -- Labels for the inputs

    Returns:
        loss -- loss of the data passed into it
    """
    activated_results = self.calculate_feed_forward(inputs, self.params)
    loss = crossentropy_loss(labels,activated_results)
    return loss
    