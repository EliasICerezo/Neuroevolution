import matplotlib.pyplot as plt
import pandas as pd
import os
import statistics

if __name__ == "__main__":
    directory = input("Directory to generate the graphics for: ")
    es_df = None
    ga_df = None
    sa_df = None
    for filename in os.listdir(directory):
      if filename.startswith('es'):
        es_df = pd.read_csv(os.path.join(directory,filename))
      elif filename.startswith('ga'):
        ga_df = pd.read_csv(os.path.join(directory,filename))
      elif filename.startswith('sa'):
        sa_df = pd.read_csv(os.path.join(directory,filename))
  
    if es_df is None or ga_df is None or sa_df is None:
      raise AttributeError("Initialization incorrect")
    
    es_df.fillna(2**32-1)
    ga_df.fillna(2**32-1)
    sa_df.fillna(2**32-1)

    # Generating the GA graphic
    unique_epochs = ga_df['epoch'].unique()
    unique_datasets = ga_df['dataset'].unique()
    for selected_dataset in unique_datasets:
      is_dataset = ga_df['dataset'] == selected_dataset
      dataset_df = ga_df[is_dataset]
      max_fitness_list = []
      min_fitness_list = []
      median_fitness_list = []
      for selected_epoch in unique_epochs:
        is_epoch = dataset_df['epoch'] == selected_epoch
        epoch_df = dataset_df[is_epoch]
        max_fitness = epoch_df['max_fitness'].to_list()
        median_fitness = epoch_df['median_fitness'].to_list()
        min_fitness = epoch_df['min_fitness'].to_list()
        max_fitness_list.append(statistics.mean(max_fitness))
        median_fitness_list.append(statistics.mean(median_fitness))
        min_fitness_list.append(statistics.mean(min_fitness))
      x = [i for i in range(int(max(unique_epochs))+1)]
      plt.plot(x,max_fitness_list, label='Max Fitness')
      plt.plot(x,min_fitness_list, label='Min Fitness')
      plt.plot(x,median_fitness_list, label='Median Fitness')
      plt.xlabel('Epochs')
      plt.ylabel('GA Fitness')
      plt.title('GA Fitness for {} {}'.format(directory, selected_dataset))
      plt.legend()
      plt.savefig('GA_figure_{}_{}'.format(directory, selected_dataset))
      plt.cla()
    # Generating the multigraphic
    for selected_dataset in unique_datasets:
      is_sa_dataset = sa_df['dataset'] == selected_dataset
      is_es_dataset = es_df['dataset'] == selected_dataset
      dataset_sa_df = sa_df[is_sa_dataset]
      dataset_es_df = es_df[is_es_dataset]
      sa_list = []
      es_list = []
      for selected_epoch in unique_epochs:
        is_sa_epoch = dataset_sa_df['epoch'] == selected_epoch
        is_es_epoch = dataset_es_df['epoch'] == selected_epoch
        epoch_sa_df = dataset_sa_df[is_sa_epoch]
        epoch_es_df = dataset_es_df[is_es_epoch]
        sa_fitness = epoch_sa_df['fitness'].to_list()
        es_fitness = epoch_es_df['fitness'].to_list()
        sa_list.append(statistics.mean(sa_fitness))
        es_list.append(statistics.mean(es_fitness))
      plt.plot(x,sa_list, label="Simulated Annealing Fitness")
      plt.plot(x,es_list, label="Evolutionary Strategy Fitness")
      plt.plot(x,min_fitness_list, label="Genetic Algorithm Fitness")
      plt.xlabel('Epochs')
      plt.ylabel('Fitness')
      plt.title('Fitness for {} {}'.format(directory, selected_dataset))
      plt.legend()
      plt.savefig('Fitness_figure_{}_{}'.format(directory, selected_dataset))
      plt.cla()