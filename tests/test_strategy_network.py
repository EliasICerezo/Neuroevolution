from neuroevolution.networks.strategy_neural_network import StrategyNeuralNetwork
import pytest

class TestStrategyNN:
  def test_basic_construction_initializes_at_least_one_member_population(self):
    nnet = StrategyNeuralNetwork([3,1], 1, 3)
    assert 'P0' in nnet.population.keys()

  def test_basic_construction_initializes_the_number_of_individuals_correctly(self):
    pop_size = 50
    nnet = StrategyNeuralNetwork([3,1],1,3,None,pop_size)
    assert len(nnet.population.keys()) == pop_size
    
  def test_basic_construction_raises_error_when_number_of_classes_is_incorrect(self):
    pop_size = 10
    with pytest.raises(AttributeError):
      _ = StrategyNeuralNetwork([3,1],2,3,None,pop_size)
    
  def test_basic_construction_raises_error_when_number_of_inputs_is_incorrect(self):
    pop_size = 10
    with pytest.raises(AttributeError):
      _ = StrategyNeuralNetwork([3,1],1,4,None,pop_size)
    
  def test_basic_construction_raises_error_when_no_layers_provided(self):
    pop_size = 10
    with pytest.raises(AttributeError):
      _ = StrategyNeuralNetwork([],1,4,None,pop_size)
  
  def test_basic_construction_raises_error_with_0_individuals(self):
    pop_size = 0
    with pytest.raises(AttributeError):
      _ = StrategyNeuralNetwork([3,1],1,4,None,pop_size)