from neuroevolution.basic_neural_network import BasicNeuralNetwork
import pytest
import pytest_mock
import numpy as np



class TestBasicNN:
  
  def test_empty_construction_has_a_greater_than_zero_lr(self):
    a = np.array([[1]])
    nnet = BasicNeuralNetwork(a,a)
    assert nnet.learning_rate > 0
  
  def test_construction_without_matching_w_n_b_raises_exception(self):
    w = [1]
    b = [0]
    with pytest.raises(AttributeError):
      nnet = BasicNeuralNetwork(w,b)
  
  def test_construction_with_weights_has_a_bias_assigned(self):
    a = np.array([[1]])
    b = np.array([[1]])
    nnet = BasicNeuralNetwork(a,b)
    assert len(nnet.weights) > 0
    assert len(nnet.biases) > 0
  
  def test_train(self, mocker: pytest_mock.mocker):
    expected_feed_forward = np.ndarray((3,5))
    expected_delta = np.ndarray((1,5))
    feed_forward_mocker = mocker.patch(
        'neuroevolution.basic_neural_network.BasicNeuralNetwork.feed_forward',
        return_value=expected_feed_forward)
    backpropagation_mocker = mocker.patch(
        'neuroevolution.basic_neural_network.BasicNeuralNetwork.backpropagation',
        return_value=expected_feed_forward)
    weight_updating_mocker = mocker.patch(
        'neuroevolution.basic_neural_network.BasicNeuralNetwork.weight_updating')
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    weights = np.random.rand(3,1)
    bias = np.random.rand(1)
    nnet = BasicNeuralNetwork(weights, bias)
    nnet.train(feature_set, labels, 1)
    feed_forward_mocker.assert_called_once_with(feature_set)
    backpropagation_mocker.assert_called_once_with(expected_feed_forward, labels)
    # weight_updating_mocker.assert_called_once_with(expected_delta, feature_set)

  

  
  

  


  