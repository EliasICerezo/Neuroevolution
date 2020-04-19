from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
import pytest
import pytest_mock
import numpy as np
import copy

class TestBasicNN:
  
  def test_basic_construction_has_a_predefined_learning_rate(self):
    nnet = BasicNeuralNetwork([3,1], 1, 3)
    assert nnet.learning_rate > 0
  
  def test_construction_without_matching_input_size_and_nnet_structure(self):
    with pytest.raises(AttributeError):
      # If you're expecting to get an exception, you don't have to save
      # object reference.
      BasicNeuralNetwork([3,1],1,5)
  
  def test_construction_with_not_matching_output_and_num_classes_raises_exception(self):
    with pytest.raises(AttributeError):
      BasicNeuralNetwork([3,1],2,3)
  
  def test_construction_initializes_weights_and_biases(self):
    nnet = BasicNeuralNetwork([3,1],1,3)
    # This network has only one layer comunicating the input with the output, so
    # it only contains a weight array 'W1' and bias one 'b1'
    assert nnet.params['W1'] is not None
    assert nnet.params['b1'] is not None
  
  def test_calculate_feed_forward_return_is_valid(self):
    # In this test we will calculate the forward propagation with the sigmoid
    # activation function, which should return a value between 0 and 1
     nnet = BasicNeuralNetwork([3,1],1,3)
     feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
     activated_results = nnet.feed_forward(feature_set)
     assert (activated_results > 0.0).all() 
     assert (activated_results < 1.0).all()
  
  def test_train_with_0_epochs_does_nothing(self):
    nnet = BasicNeuralNetwork([3,1],1,3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    params_copy = copy.deepcopy(nnet.params) 
    nnet.train(feature_set,labels, 0)
    assert str(nnet.params) == str(params_copy)
  
  def test_backpropgation_creates_all_the_derivatives_required(self):
    nnet = BasicNeuralNetwork([3,1],1,3)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    # We need to use a vector the shape of the transposed matrix of the labels
    nnet.backpropagation(feature_set, labels, np.random.randn(5,1))
    assert 'dl_wrt_w1' in nnet.params.keys()
    assert 'dl_wrt_b1' in nnet.params.keys()

  

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
  
  def test_feed_forward_with_more_than_two_layers(self):
    #  In this test we will calculate the forward propagation with the sigmoid
    # activation function, which should return a value between 0 and 1
     nnet = BasicNeuralNetwork([3,2,1],1,3)
     feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
     activated_results = nnet.feed_forward(feature_set)
     assert (activated_results > 0.0).all() 
     assert (activated_results < 1.0).all()    
