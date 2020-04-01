from activation_functions import sigmoid, sigmoid_der
from error_functions import MSE
class BasicNeuralNetwork:
  def __init__(self, layers = [], biases = [] ):
    self.layers = layers
    self.biases = biases

  def add_layer(self, layer, bias):
    self.layers.append(layer)
    self.biases.append(bias)
    if len(self.layers) != len(self.biases):
      print("Lengths are not coincident")

  def train(self, inputs, targets):
    output = -1
    output = self.feed_forward(inputs, output)
    self.backpropagation(output, targets)
    print(output)

  def feed_forward(self, inputs, result):
    weighted_input = 0
    for i in range(0, len(self.layers)):
      # If the inputs are compatible with the input layer
      if len(inputs) == len(self.layers[0]):
        for (j, neuron) in enumerate(self.layers[i], start=0):
          if i == 0:
            weighted_input += neuron.calculate_output(inputs[j])
            if j == len(self.layers[i])-1:
              weighted_input += self.biases[i]
              weighted_input = sigmoid(weighted_input)
          elif i == len(self.layers)-1:
            result = neuron.calculate_output(weighted_input)
            result = sigmoid(result)
      else:
        print("Input length ({}) is not correspondant to the number of input neurons: {}".format(len(inputs), len(self.layers[0]) ) )
        break
    return [result]
  
  def backpropagation(self, outputs, targets):
    #  We calculate the error made by the nnet
    error = MSE(outputs, targets)
    
    print(error)
    pass
 

