import pytest
from basic_neural_network import BasicNeuralNetwork

class TestBasicNN:
  # Tets for construction of the nn
  def test_empty_construction_is_not_none(self):
    nnet = BasicNeuralNetwork()
    assert nnet
  
  def test_empty_construction_has_empty_weights(self):
    nnet = BasicNeuralNetwork()
    assert len(nnet.weights)==0
  
  def test_empty_construction_has_empty_biases(self):
    nnet = BasicNeuralNetwork()
    assert len(nnet.biases)==0
  
  def test_empty_construction_has_a_greater_than_zero_lr(self):
    nnet = BasicNeuralNetwork()
    assert nnet.learning_rate > 0
  
  def test_construction_without_matching_w_n_b_raises_exception(self):
    w = [1]
    b = []
    with pytest.raises(AttributeError):
      nnet = BasicNeuralNetwork(w,b)
  
  def test_construction_with_weights_has_a_bias_assigned(self):
    w = [1]
    b = [0]
    nnet = BasicNeuralNetwork(w,b)
    assert len(nnet.weights) > 0
    assert len(nnet.biases) > 0
  
  
  
  
  

  


  