import pytest
from neuroevolution.activation_functions import sigmoid, sigmoid_der, relu, relu_der

class TestActivationFunction:
  def test_sigmoid_of_zero_is_point_five(self):
    assert sigmoid(0) == 0.5

  def test_sigmoid_derivative_with_zero_is_point_25(self):
    assert sigmoid_der(0) == 0.25
  
  def test_of_relu_less_than_zero_is_zero(self):
    """Testing the non-linearity of the function with these 2 tests
    """
    assert relu(-1) == 0
  
  def test_relu_x_greater_than_zero_returns_x(self):
    """Testing the non-linearity of the function with these 2 tests
    """
    assert relu(1)==1
  
  def test_relu_der_zero(self):
    assert relu_der(0) == 0
  
  def test_relu_der_one(self):
    assert relu_der(1) == 1
  
  
    