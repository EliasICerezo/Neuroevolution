import pytest
from error_functions import MSE


class TestErrorFunctions:

  def test_MSE_of_zero_is_zero(self):
    assert MSE([1],[1]) == 0

  def test_MSE_of_one_is_one(self):
    assert MSE([0],[1]) == 1
  
  def test_MSE_multiple_values(self):
    assert MSE([0,1,2,3,4],[1,2,3,4,5]) == 1
  