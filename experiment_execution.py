from neuroevolution.networks.basic_neural_network import BasicNeuralNetwork
from neuroevolution.networks.genetic_neural_network import GeneticNeuralNetwork
from neuroevolution.networks.annealed_neural_network import AnnealedNeuralNetwork
from neuroevolution.networks.strategy_neural_network import StrategyNeuralNetwork
from neuroevolution.networks.random_search_nn import RandomSearchNeuralNetwork
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import dataset_manipulation.helpers as helpers
import typing
import pandas as pd
import numpy as np
import os
import time
import threading

PRIMES = [3,17,101,5003,70001,600011,1234577,98765441,198765433,1928765459,
          11,1861,257,49297,849347,1849283,71849363,731849389,26591,40879,
          3731849309,7,88191,17191,27791,12347,53819,51341,99089,37643]

df = pd.DataFrame(columns = ['dataset','neural_network','training_loss',
      'testing_loss', 'time', 'number_of_folds'])
ga_df = pd.DataFrame(columns = ['dataset','seed','epoch','max_fitness',
      'median_fitness', 'min_fitness'])
sa_df = pd.DataFrame(columns = ['dataset','seed','epoch','fitness'])
es_df = pd.DataFrame(columns = ['dataset','seed','epoch','fitness'])

resource_lock = threading.Lock()

def prepare_dataset(csvname:str, transform_list:typing.List[str],
    drop_list: typing.List[str] = [], labels_id:str = 'y'):
  """Method that process the datasets and prepares them to be used in the
  different neural networks

  Arguments:
      csvname {str} -- CSV path to be preprocessed
      transform_list {typing.List[str]} -- List of labels of the dataframe to be
      transformed.

  Keyword Arguments:
      drop_list {typing.List[str]} -- List of labels to be dropped of the
      dataframe (default: {[]}).
      labels_id {str} -- Label of the column of the dataframe to be selected
      as labels od the neural network (default: {'y'}).

  Returns:
      inputs,labels -- Inputs and labels extracted of the dataframe.
  """
  if not os.path.isfile(csvname):
    raise AttributeError("CSV path provided is not a file")
  df = pd.read_csv(csvname)
  df, tlist = helpers.transform_alpha_list_into_numeric(df,transform_list)
  drop_list = drop_list + tlist
  return helpers.extract_inputs_and_labels(df,labels_id,drop_list)



def init_datasets():
  """Method that loads the datasets in memory

  Returns:
      list,list -- Label list and inputs list
  """
  inputs_list = []
  labels_list = []
  i,lab = prepare_dataset('datasets/iris.csv', ['y'], [], 'new_y')
  inputs_list.append(i)
  labels_list.append(lab)
  i,lab = prepare_dataset('datasets/wine.csv', ['y'], ['price'], 'new_y')
  inputs_list.append(i)
  labels_list.append(lab)
  i,lab = prepare_dataset('datasets/breast-cancer-wisconsin.csv', ['y'], ['sample_number'], 'new_y')
  inputs_list.append(i)
  labels_list.append(lab)
  i,lab = prepare_dataset('datasets/processed.cleveland.csv', ['y'], [], 'new_y')
  inputs_list.append(i)
  labels_list.append(lab)
  return labels_list, inputs_list


def save_into_df(n_row:dict):
  """Method that saves into the dataframe the data collected by the neural
  network

  Arguments:
      n_row {dict} -- Row to be inserted in the dataframe

  Returns:
      dataframe -- Updated dataframe
  """
  resource_lock.acquire()
  try:
    global df
    df = df.append(n_row, ignore_index=True)
  finally:
    resource_lock.release()
  return df


def basic_nn_tenant():
  """Basic Neural Network tenant to be executed concurrently
  """
  nnet = BasicNeuralNetwork([tr_data.shape[1], 5, tr_labels.shape[1]], tr_labels.shape[1], tr_data.shape[1])
  init_t = time.time()
  loss = nnet.train(tr_data, tr_labels, num_epochs)
  t = time.time() - init_t
  te_loss = nnet.test(te_data, te_labels)
  n_row = {'dataset': datasets[i], 'neural_network':'BasicNN',
      'training_loss':loss, 'testing_loss':te_loss, 'time':t, 'number_of_folds': number_of_folds, 'epochs': num_epochs, 'fold': idx}
  save_into_df(n_row)


def genetic_nn_tenant():
  """Genetic Neural Network tenant to be executed concurrently
  """
  nnet = GeneticNeuralNetwork([tr_data.shape[1], 5, tr_labels.shape[1]], tr_labels.shape[1], tr_data.shape[1])
  init_t = time.time()
  loss, ga_statistics = nnet.train(tr_data, tr_labels, num_epochs)
  t = time.time() - init_t
  te_loss = nnet.test(te_data, te_labels)
  n_row = {'dataset': datasets[i], 'neural_network':'GeneticNN',
      'training_loss':loss, 'testing_loss':te_loss, 'time':t, 'number_of_folds': number_of_folds, 'epochs': num_epochs, 'fold': idx}
  save_into_df(n_row)
  ga_statistics['seed'] = r
  ga_statistics['dataset'] = datasets[i]
  global ga_df
  ga_df = ga_df.append(ga_statistics, ignore_index=True)


def strategy_nn_tenant():
  """Strategy Neural Network tenant to be executed concurrently
  """
  nnet = StrategyNeuralNetwork([tr_data.shape[1], 5, tr_labels.shape[1]], tr_labels.shape[1], tr_data.shape[1], verbose=False)
  init_t = time.time()
  loss, es_statistics = nnet.train(tr_data, tr_labels, num_epochs)
  t = time.time() - init_t
  te_loss = nnet.test(te_data, te_labels)
  n_row = {'dataset': datasets[i], 'neural_network':'StrategyNN',
      'training_loss':loss, 'testing_loss':te_loss, 'time':t, 'number_of_folds': number_of_folds, 'epochs': num_epochs, 'fold': idx}
  save_into_df(n_row)
  es_statistics['seed'] = r
  es_statistics['dataset'] = datasets[i]
  global es_df
  es_df = es_df.append(es_statistics, ignore_index=True)


def random_nn_tenant():
  """Random Neural Network tenant to be executed concurrently
  """
  nnet = RandomSearchNeuralNetwork([tr_data.shape[1], 5, tr_labels.shape[1]], tr_labels.shape[1], tr_data.shape[1])
  init_t = time.time()
  loss = nnet.train(tr_data, tr_labels, num_epochs)
  t = time.time() - init_t
  te_loss = nnet.test(te_data, te_labels)
  n_row = {'dataset': datasets[i], 'neural_network':'RandomNN',
      'training_loss':loss, 'testing_loss':te_loss, 'time':t, 'number_of_folds': number_of_folds, 'epochs': num_epochs, 'fold': idx}
  save_into_df(n_row)

def annealed_nn_tenant():
  """Simulated Annealing Neural Network tenant to be executed concurrently
  """
  nnet = AnnealedNeuralNetwork([tr_data.shape[1], 5, tr_labels.shape[1]], tr_labels.shape[1], tr_data.shape[1])
  init_t = time.time()
  loss, sa_statistics= nnet.train(tr_data, tr_labels, num_epochs)
  t = time.time() - init_t
  te_loss = nnet.test(te_data, te_labels)
  n_row = {'dataset': datasets[i], 'neural_network':'AnnealedNN',
      'training_loss':loss, 'testing_loss':te_loss, 'time':t, 'number_of_folds': number_of_folds, 'epochs': num_epochs, 'fold': idx}
  save_into_df(n_row)
  sa_statistics['seed'] = r
  sa_statistics['dataset'] = datasets[i]
  global sa_df
  sa_df = sa_df.append(sa_statistics, ignore_index=True)

if __name__ == "__main__":
  
  number_of_folds = 5
  num_epochs = 5
  labels_list, inputs_list = init_datasets()
  datasets = ['iris', 'wine', 'breast_cancer', 'heart']
  dfidx = 0
  for r in PRIMES:
    np.random.seed = r

    for i in range(len(labels_list)):
      labels = to_categorical(labels_list[i])
      inputs = inputs_list[i]
      kf = KFold(number_of_folds,shuffle=True, random_state=r)
      tr_gen = kf.split(inputs,labels)
      tr = []
      te = []
      for training,testing in tr_gen:
        tr.append(training)
        te.append(testing)
        # idx = np.random.randint(0,number_of_folds)
      for idx in range(len(tr)):
        print(idx)
        tr_idx, te_idx = (tr[idx],te[idx])
        tr_data = inputs[tr_idx]
        tr_labels = labels[tr_idx]
        te_data = inputs[te_idx]
        te_labels = labels[te_idx]
        t1 = threading.Thread(target=basic_nn_tenant)
        t2 = threading.Thread(target=genetic_nn_tenant)
        t3 = threading.Thread(target=strategy_nn_tenant)
        t4 = threading.Thread(target=random_nn_tenant)
        t5 = threading.Thread(target=annealed_nn_tenant)
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t5.daemon = True
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        print(df)
    df.to_csv("results{}(5-2-1).csv".format(str(num_epochs)), index=False)
    ga_df.to_csv("ga{}(5-2-1).csv".format(str(num_epochs)), index=False)
    es_df.to_csv("es{}(5-2-1).csv".format(str(num_epochs)), index=False)
    sa_df.to_csv("sa{}(5-2-1).csv".format(str(num_epochs)), index=False)
  

