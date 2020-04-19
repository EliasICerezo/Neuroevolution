import pytest
from neuroevolution import activation_functions
# It's always better to import package than methods directly.

class TestActivationFunction:
  def test_sigmoid_of_zero_is_point_five(self):
    assert activation_functions.sigmoid(0) == 0.5

  def test_sigmoid_derivative_with_zero_is_point_25(self):
    assert activation_functions.sigmoid_der(0) == 0.25
  
  def test_of_relu_less_than_zero_is_zero(self):
    """Testing the non-linearity of the function with these 2 tests
    """
    assert activation_functions.relu(-1) == 0
  
  def test_relu_x_greater_than_zero_returns_x(self):
    """Testing the non-linearity of the function with these 2 tests
    """
    assert activation_functions.relu(1)==1
  
  def test_relu_der_zero(self):
    assert activation_functions.relu_der(0) == 0
  
  def test_relu_der_one(self):
    assert activation_functions.relu_der(1) == 1
