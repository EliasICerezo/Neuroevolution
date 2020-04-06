from neuroevolution.basic_neural_network import BasicNeuralNetwork
import pytest
import pytest_mock
import numpy as np

class TestBasicNN:
  
  def test_basic_construction_has_a_predefined_learning_rate(self):
    nnet = BasicNeuralNetwork([3,1], 1, 3)
    assert nnet.learning_rate > 0
  
  def test_construction_without_matching_input_size_and_nnet_structure(self):
    with pytest.raises(AttributeError):
      nnet = BasicNeuralNetwork([3,1],1,5)
  
  def test_construction_with_not_matching_output_and_num_classes_raises_exception(self):
    with pytest.raises(AttributeError):
      nnet = BasicNeuralNetwork([3,1],2,3)
  
  def test_basic_nn_reduces_loss_after_training(self):
    nnet = BasicNeuralNetwork([3,1], 1, 3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    nnet.train(feature_set, labels, 1000)
    assert nnet.loss[-1] < nnet.loss[0]

  def test_basic_training_modifies_weights(self):
    nnet = BasicNeuralNetwork([3,1], 1, 3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    init_weights = np.copy(nnet.params['W1'])
    nnet.train(feature_set,labels,1000)
    comparison = init_weights != nnet.params['W1']
    assert comparison.all()

  def test_basic_training_adds_same_number_of_loss_than_epochs(self):
    nnet = BasicNeuralNetwork([3,1], 1, 3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    num_epochs = 100
    nnet.train(feature_set,labels,num_epochs)
    assert len(nnet.loss) == num_epochs

  

  # def test_train(self, mocker: pytest_mock.mocker):
  #   expected_feed_forward = np.ndarray((3,5))
  #   expected_delta = np.ndarray((1,5))
  #   feed_forward_mocker = mocker.patch(
  #       'neuroevolution.basic_neural_network.BasicNeuralNetwork.feed_forward',
  #       return_value=expected_feed_forward)
  #   backpropagation_mocker = mocker.patch(
  #       'neuroevolution.basic_neural_network.BasicNeuralNetwork.backpropagation',
  #       return_value=expected_feed_forward)
  #   weight_updating_mocker = mocker.patch(
  #       'neuroevolution.basic_neural_network.BasicNeuralNetwork.weight_updating')
  #   feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
  #   labels = np.array([[1,0,0,1,1]])
  #   labels = labels.reshape(5,1)
  #   weights = np.random.rand(3,1)
  #   bias = np.random.rand(1)
  #   nnet = BasicNeuralNetwork([3,1], 1, 3)
  #   nnet.train(feature_set, labels, 10)
  #   feed_forward_mocker.assert_called_once_with(feature_set)
  #   backpropagation_mocker.assert_called_once_with(expected_feed_forward, labels)
    # weight_updating_mocker.assert_called_once_with(expected_delta, feature_set)

  

  
  

  


  