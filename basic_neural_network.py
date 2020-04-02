from activation_functions import sigmoid, sigmoid_der
from error_functions import MSE
import numpy as np
class BasicNeuralNetwork:
  """Class that implements the basic behaviour of a neural network.
  """

  def __init__(self, weights = [], biases = [], lr = 0.05 ):
    """Contructor of the basic Neural Network Object

    Keyword arguments:
      weights -- weights for the NNET to be initialized with, you can omit this
      since you can add the layers of weights later on.
      biases -- biases corresponding to the layers of weights.
      lr -- Stands for learning rate and it ts the size of the movement once you
      start training it.
    """  
    self.weights = weights
    self.biases = biases
    self.learning_rate = lr

  def add_layer(self, layer, bias):
    """Method that allows to add a new layer to the existing ones.
    You can create an empty neural network and after that build it using this 
    method.
    KeyWord Arguments:
      layer -- the list of weights to be added to the main structure
      bias -- the bias that correspond to the layer we are adding
    """  
    self.weights.append(layer)
    self.biases.append(bias)
    if len(self.weights) != len(self.biases):
      print("Lengths are not coincident")
  
  def train(self, inputs, targets, epochs):
    """Training method of the neural network.
    It optimized the weights for a given input
    Keyword Arguments:
      inputs -- the features to be predicted by the neural network
      targets -- the targets that match the features for this predictions.
      epochs -- number of iterations of optimization that the neural network will
      perform
    """  
    print(self.weights)
    for i in range(epochs):
      forward_pass = self.feed_forward(inputs)
      delta = self.backpropagation(forward_pass, targets)
      features = inputs.T
      self.weight_updating(delta, features)
      # if i % 100 == 0:
      #   print("Epoch {}".format(i))
    print(self.weights)

  def feed_forward(self, inputs):
    """Function that performs the forward pass through the neural network, it
    computes the dot product between the features and the weights and then adds
    the bias assigned to that layer.
    Returns:
        activated_result -- The result of the dot product and the activation
        function
    """  
    forward_pass = np.dot(inputs, self.weights) + self.biases[0]
    activated_result = sigmoid(forward_pass)
    return activated_result
  
  def backpropagation(self, forward_pass, labels):
    """The backward pass to update the weights.
    It computes the ammount in which the weights must be changed to match the 
    outputs by the calculation of the error that the neural network is making
    at the moment.
    
    Returns:
        z_delta -- The delta diference in which the weights should be modified
    """  
    #  We calculate the error made by the nnet
    error = forward_pass - labels
    # We use the derivative of the sigmoid to extract the correct data
    dcost_dpred = error
    dpred_dz = sigmoid_der(forward_pass)
    z_delta = dcost_dpred * dpred_dz
    return z_delta

  def weight_updating(self, delta, inputs):
    self.weights -= self.learning_rate * np.dot(inputs, delta)
    for num in delta:
      self.biases[0] -= self.learning_rate * num

    
 

