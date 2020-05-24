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
import time

PRIMES = [3,17,101,5003,70001,600011,1234577,98765441,198765433,1928765459,
          19728765443,179728765483,1979728765451,12979728765461,129797287625459,
          1729797287625437,11,1861,257,49297,849347,1849283,71849363,731849389,
          3731849309,13731849301,213731849351,1213731849359,7,8819]

def prepare_dataset(csvname:str, transform_list:typing.List[str],
    drop_list: typing.List[str] = [], labels_id:str = 'y'):
  if not os.path.isfile(csvname):
    raise AttributeError("CSV path provided is not a file")
  df = pd.read_csv(csvname)
  df, tlist = helpers.transform_alpha_list_into_numeric(df,transform_list)
  drop_list = drop_list + tlist
  return helpers.extract_inputs_and_labels(df,labels_id,drop_list)



def init_datasets():
  inputs_list = []
  labels_list = []
  i,lab = prepare_dataset('datasets/processed.cleveland.csv', [], [], 'y')
  inputs_list.append(i)
  labels_list.append(lab)
  i,lab = prepare_dataset('datasets/iris.csv', ['y'], [], 'new_y')
  inputs_list.append(i)
  labels_list.append(lab)
  i,lab = prepare_dataset('datasets/breast-cancer-wisconsin.csv', [], ['sample_number'], 'y')
  inputs_list.append(i)
  labels_list.append(lab)
  i,lab = prepare_dataset('datasets/wine.csv', [], ['price'], 'y')
  inputs_list.append(i)
  labels_list.append(lab)
  return labels_list, inputs_list

if __name__ == "__main__":
    num_epochs = 10
    labels_list, inputs_list = init_datasets()
    df = pd.DataFrame(columns = ['dataset','neural_network','training_loss', 'time', 'number_of_folds'])
    datasets = ['heart', 'iris', 'breast_cancer', 'wine']
    dfidx = 0
    for r in PRIMES:
      np.random.seed = r
      for i in range(len(labels_list)):
        labels = labels_list[i]
        inputs = inputs_list[i]
        nnet = BasicNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
        init_t = time.time()
        loss = nnet.train(inputs, labels, num_epochs)
        t = time.time() - init_t
        n_row = {'dataset': datasets[i], 'neural_network':'BasicNN',
            'training_loss':loss, 'time':t, 'number_of_folds': 1}
        df = df.append(n_row, ignore_index=True)
        
        nnet = GeneticNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
        init_t = time.time()
        loss = nnet.train(inputs, labels, num_epochs)
        t = time.time() - init_t
        n_row = {'dataset': datasets[i], 'neural_network':'GeneticNN',
            'training_loss':loss, 'time':t, 'number_of_folds': 1}
        df = df.append(n_row, ignore_index=True)

        nnet = StrategyNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
        init_t = time.time()
        loss = nnet.train(inputs, labels, num_epochs)
        t = time.time() - init_t
        n_row = {'dataset': datasets[i], 'neural_network':'StrategyNN',
            'training_loss':loss, 'time':t, 'number_of_folds': 1}
        df = df.append(n_row, ignore_index=True)
        
        nnet = RandomSearchNeuralNetwork([inputs.shape[1], 10, 1], 1, inputs.shape[1])
        init_t = time.time()
        loss = nnet.train(inputs, labels, num_epochs)
        t = time.time() - init_t
        n_row = {'dataset': datasets[i], 'neural_network':'RandomNN',
            'training_loss':loss, 'time':t, 'number_of_folds': 1}
        df = df.append(n_row, ignore_index=True)
        print("Inner Loop iteration: {}".format(i))
    breakpoint()
