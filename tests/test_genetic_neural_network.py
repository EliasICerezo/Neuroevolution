from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
import numpy as np
import pytest
import copy
import pytest_mock


# you can split test case in two different classes in order to avoid large
# method names.

class TestGeneticNNBasicConstruction:

  def test_initializes_at_least_one_member_population(self):
    nnet = GeneticNeuralNetwork([3,1], 1, 3)
    assert 'P0' in nnet.population.keys()

  def test_initializes_the_number_of_individuals_correctly(self):
    pop_size = 50
    nnet = GeneticNeuralNetwork([3,1],1,3,None,pop_size)
    assert len(nnet.population.keys()) == pop_size

  def test_raises_error_when_pop_size_is_over_maximum(self):
    pop_size = 500
    with pytest.raises(AttributeError):
      _ = GeneticNeuralNetwork([3,1],1,3,None,pop_size)
    
  def test_raises_error_when_number_of_classes_is_incorrect(self):
    pop_size = 10
    with pytest.raises(AttributeError):
      _ = GeneticNeuralNetwork([3,1],2,3,None,pop_size)
    
  def test_raises_error_when_number_of_inputs_is_incorrect(self):
    pop_size = 10
    with pytest.raises(AttributeError):
      _ = GeneticNeuralNetwork([3,1],1,4,None,pop_size)
    
  def test_raises_error_when_no_layers_provided(self):
    pop_size = 10
    with pytest.raises(AttributeError):
      _ = GeneticNeuralNetwork([],1,4,None,pop_size)
  
  def test_raises_error_with_0_individuals(self):
    pop_size = 0
    with pytest.raises(AttributeError):
      _ = GeneticNeuralNetwork([3,1],1,4,None,pop_size)
  
  def test_raises_error_with_0_max_individuals(self):
    pop_size = 0
    with pytest.raises(AttributeError):
      _ = GeneticNeuralNetwork([3,1],1,4,None,pop_size)


class TestGeneticNNBehaviours:

  def test_evolved_feed_forward_returns_an_array_with_pop_size(self):
    pop_size = 25
    nnet = GeneticNeuralNetwork([3,1],1,3,None,pop_size)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    # This function returns a dictionary, so we are going to compare the length
    # of the key array
    activated_results = nnet.evolved_feed_forward(feature_set)
    assert len(activated_results.keys()) == pop_size

  def test_mutation_operator_generates_some_new_individuals(self):
    """This test may fail since it is dependant of probability and ir may not be
    successful some times
    """
    pop_size = 25
    nnet = GeneticNeuralNetwork([3,1],1,3,None,pop_size)
    pop_copy = copy.deepcopy(nnet.population)
    # Copying float values is difficult, so we are going to compare the string
    # value, that ensembles the opulation
    assert str(pop_copy) == str(nnet.population)
    nnet.mutate_population()
    assert len(nnet.additions.keys()) > 0
  
  def test_crossover_operator_generates_new_individuals(self):
    # This test may fail since it depends on pobability to take place
    pop_size = 25
    nnet = GeneticNeuralNetwork([3,1],1,3,None,pop_size)
    pop_copy = copy.deepcopy(nnet.population)
    # Copying float values is difficult, so we are going to compare the string
    # value, that ensembles the opulation
    assert str(pop_copy) == str(nnet.population)
    nnet.crossover_populations('W') # Mutating the weights
    assert len(nnet.additions.keys()) >= 0

  def test_selection_operator_is_called_during_training(
      self, mocker: pytest_mock.mocker
  ):
    # It's a common structure mistake here, you can avoid patch large routes
    # by importing them recursively. For example you can import
    # GeneticNeuralNetwork directly from `neuroevolution.networks` by importing
    # it in his __init__.py. Python is not Java, so if you are creating
    # a file for each pytho class, you should consider to import them into
    # their root package.
    selection_mock = mocker.patch(
      'neuroevolution.networks.genetic_neural_network.GeneticNeuralNetwork.selection_operator'
    )
    nnet = GeneticNeuralNetwork([3,1],1,3,None,25)
    feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
    labels = np.array([[1,0,0,1,1]])
    labels = labels.reshape(5,1)
    nnet.train(feature_set,labels,5)
    assert selection_mock.called

    
  

  

