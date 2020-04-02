import pytest
from activation_functions import sigmoid

class TestActivationFunction:
  def test_sigmoid_of_zero_is_point_five(self):
    assert sigmoid(0) == 0.5
    