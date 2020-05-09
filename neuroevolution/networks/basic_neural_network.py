from neuroevolution.activation_functions import sigmoid, sigmoid_der, relu, relu_der
from neuroevolution.error_functions import MSE, crossentropy_loss
import numpy as np
import typing
class BasicNeuralNetwork:
  """Class that implements the basic behaviour of a neural network.
  It represents the fully connected networks, so take into account that all the
  links in between nodes will be found here.
  It ensembles all the basics and it is generalized to match every architecure.
  However it is important to know that since it will need to store derivatives
  to calculate the backpropagation step, it may not be super eficient in memory
  """

  def __init__(self, layers:list, num_of_classes:int, input_size:int, lr = 0.05,
               activation_functs = None):
    """Contructor of the basic Neural Network Object

    Keyword arguments:
      layers -- List of numbers that ensemble the number of neurons od each layer
      lr -- Stands for learning rate and it ts the size of the movement once you
      start training it.
    """ 
    self.layers = layers
    self.learning_rate = lr
    self.params = {}
    self.loss = []
    if layers[-1] != num_of_classes:
      raise AttributeError("The number of classes should match the last layer")
    if layers[0] != input_size:
      raise AttributeError("The input size should match the 1st layer")
    if len(self.layers) == 0:
      raise AttributeError("Can't create a neural net without a single layer ")
    self.activation_functs = activation_functs
    if activation_functs is None:
      self.activation_functs = [sigmoid for i in range(len(self.layers))]
    
    self.initialize_weithts_and_biases(self.params)
  
  def initialize_weithts_and_biases(self, store:dict):
    """Function that initialize weights and biases randomly. The random values
    are not uniformly distributed which can cause a slower convergence.
    
    Keyword Arguments:
        store {dict} -- Dictionary to put the initialization if provided (default: {None})
    
    Returns:
        [type] -- [description]
    """
    np.random.seed = 42
    initialization = {}
    for i,e in enumerate(self.layers):
      if i < len(self.layers)-1:
        initialization[
            'W{}'.format(i+1)] = np.random.randn(self.layers[i], self.layers[i+1])
        initialization[
            'b{}'.format(i+1)] = np.random.randn(self.layers[i+1])
    store.update(initialization)

  def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
    """Training method of the neural network.
    It optimized the weights for a given input
    Keyword Arguments:
      inputs -- the features to be predicted by the neural network
      targets -- the targets that match the features for this predictions.
      epochs -- number of iterations of optimization that the neural network
      will perform
    """  
    for _ in range(epochs):
      y_hat = self.feed_forward(inputs)
      loss = crossentropy_loss(targets,y_hat)
      self.loss.append(loss)
      self.backpropagation(inputs,y=targets,y_hat=y_hat)
      self.__weight_updating()


  def feed_forward(self, inputs):
    """Function that performs the forward pass through the neural network, it
    computes the dot product between the features and the weights and then adds
    the bias assigned to that layer.
    Returns:
        activated_result -- The result of the dot product and the activation
        function
    """ 
    return self.calculate_feed_forward(inputs,self.params)
  
  def calculate_feed_forward(self, inputs, store):
    for i in range(len(self.layers)-1):
        if i == 0:
          Z_i = inputs.dot(store['W{}'.format(i+1)]) + store['b{}'.format(i+1)]
          A_i = self.activation_functs[i](Z_i)
        else:
          Z_i = A_i.dot(store['W{}'.format(i+1)]) + store['b{}'.format(i+1)]
          A_i = self.activation_functs[i](Z_i)
        store['Z{}'.format(i+1)] = Z_i 
        store['A{}'.format(i+1)] = A_i  
    y_hat = A_i
    return y_hat

  def backpropagation(self, inputs, y, y_hat):
    """The backward pass to update the weights.
    It computes the ammount in which the weights must be changed to match the 
    outputs by the calculation of the error that the neural network is making
    at the moment.
    Note: It only performs the backpropagation, not the weight updating.
    
    Returns:
        z_delta -- The delta diference in which the weights should be modified
    """  
    dl_wrt_yhat = -(np.divide(y, y_hat) - np.divide((1 - y),(1-y_hat)))
    dl_wrt_sig = y_hat * (1-y_hat)
    dl_wrt_z_i = dl_wrt_yhat * dl_wrt_sig
    for i in range(len(self.layers)-1,0,-1):
      if i != len(self.layers)-1:
        # TODO: Make this derivative general to all activation functions
        dl_wrt_z_i = dl_wrt_A_j * sigmoid_der(self.params['Z{}'.format(i)])
      if i > 1:
        dl_wrt_A_j = dl_wrt_z_i.dot(self.params['W{}'.format(i)].T)
        dl_wrt_w_i = self.params['A{}'.format(i-1)].T.dot(dl_wrt_z_i)
      else:
        dl_wrt_w_i = inputs.T.dot(dl_wrt_z_i)
      dl_wrt_b_i = np.sum(dl_wrt_z_i, axis=0)
      self.params['dl_wrt_w{}'.format(i)] = dl_wrt_w_i
      self.params['dl_wrt_b{}'.format(i)] = dl_wrt_b_i

  def __weight_updating(self):
    """In this fucnction we update the weights of the neural network using the 
    derivatives calculated in the backpropagation pass.
    """
    for i in range(len(self.layers)-1):
      self.params['W{}'.format(i+1)] = (
          self.params['W{}'.format(i+1)] -
          self.learning_rate *
          self.params['dl_wrt_w{}'.format(i+1)])
      self.params['b{}'.format(i+1)] = (
          self.params['b{}'.format(i+1)] -
          self.learning_rate *
          self.params['dl_wrt_b{}'.format(i+1)])