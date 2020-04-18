from neuroevolution.networks.random_search_nn import RandomSearchNeuralNetwork
import numpy as np
import pytest
import copy

class TestRandomNeuralNetwork:
  
  def test_train_with_0_epochs_does_nothing(self):
    nnet = RandomSearchNeuralNetwork([3,1],1,3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    params_copy = copy.deepcopy(nnet.params) 
    nnet.train(feature_set,labels, 0)
    for (k,v) in params_copy.items():
      assert (v == nnet.params[k]).all()

  def test_train_with_100_epochs_modifies_weights(self):
    nnet = RandomSearchNeuralNetwork([3,1],1,3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    params_copy = copy.deepcopy(nnet.params) 
    nnet.train(feature_set,labels, 100)
    for (k,v) in params_copy.items():
      assert (v != nnet.params[k]).any()
  
  def test_random_network_improves_over_time(self):
    nnet = RandomSearchNeuralNetwork([3,1], 1, 3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    nnet.train(feature_set, labels, 1000)
    # It  could be the case that none of the points are better than the initial
    # one so, the assertion is <= rather than <
    assert nnet.loss[-1] <= nnet.loss[0]
  