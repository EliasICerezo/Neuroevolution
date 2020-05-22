from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.networks.annealed_neural_network import AnnealedNeuralNetwork
from neuroevolution.networks.strategy_neural_network import StrategyNeuralNetwork
from neuroevolution.networks.random_search_nn import RandomSearchNeuralNetwork
import dataset_manipulation.helpers as helpers
import typing
import pandas as pd
import numpy as np
import os

def prepare_dataset(csvname:str, transform_list:typing.List[str],
    drop_list: typing.List[str] = [], labels_id:str = 'y'):
  if not os.path.isfile(csvname):
    raise AttributeError("CSV path provided is not a file")
  df = pd.read_csv(csvname)
  df, tlist = helpers.transform_alpha_list_into_numeric(df,transform_list)
  drop_list = drop_list + tlist
  return helpers.extract_inputs_and_labels(df,labels_id,drop_list)


if __name__ == "__main__":
    inputs,labels = prepare_dataset('datasets/iris.csv', ['y'], [], 'new_y')
    nnet = BasicNeuralNetwork([inputs.shape[1], 4, 1], 1, inputs.shape[1])
    nnet.train(inputs, labels, 300)
    print(nnet.loss)
    nnet = GeneticNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
    nnet.train(inputs, labels, 50)
    vs = list(nnet.population.values())
    for i in vs: print(i['loss'])
    nnet = StrategyNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
    nnet.train(inputs, labels, 50)
    vs = list(nnet.population.values())
    for i in vs: print(i['loss'])
    nnet = RandomSearchNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
    nnet.train(inputs, labels, 50)
    print(nnet.loss)
