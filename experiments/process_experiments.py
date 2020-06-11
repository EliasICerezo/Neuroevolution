import pandas as pd
import statistics
import os

if __name__ == "__main__":
  processed_df = pd.DataFrame(columns=['nnet', 'dataset', 'tr_mean', 'tr_std', 'te_mean', 'te_std', 'mean_time', 'std_time'])
  csv_path = "Experiment3/experiment3.csv" #input("Results CSV to be processed: ")
  if csv_path == "":
    raise AttributeError("Path to file provided empty")
  df = pd.read_csv(csv_path)
  df = df.fillna((2**32)-1)
  unique_nns = list(df.neural_network.unique())
  unique_datasets = list(df.dataset.unique())
  for nn in unique_nns:
    is_nn = df['neural_network'] == nn
    df_nn = df[is_nn]
    for selected_dataset in unique_datasets:
      is_data_from_dataset = df_nn['dataset'] == selected_dataset
      data_from_dataset = df_nn[is_data_from_dataset]
      tr_loss = data_from_dataset['training_loss'].to_list()
      te_loss = data_from_dataset['testing_loss'].to_list()
      time = data_from_dataset['time'].to_list()
      row = {'nnet': nn, 'dataset': selected_dataset,
        'tr_mean': statistics.mean(tr_loss), 'tr_std':statistics.stdev(tr_loss),
        'te_mean': statistics.mean(te_loss), 'te_std':statistics.stdev(te_loss),
        'mean_time': statistics.mean(time), 'std_time': statistics.stdev(time)}
      processed_df = processed_df.append(row, ignore_index = True)
  processed_df = processed_df.round(decimals=4)
  processed_df.to_csv("processed_experiments/processed_{}".format(os.path.split(csv_path)[-1]), index=False)
  for dataset in unique_datasets:
    is_data_from_dataset = processed_df['dataset'] == dataset
    data_from_dataset = processed_df[is_data_from_dataset]
    data_from_dataset = data_from_dataset.drop(columns=['dataset'])
    data_from_dataset.to_csv("{}table.csv".format(dataset), index=False)
