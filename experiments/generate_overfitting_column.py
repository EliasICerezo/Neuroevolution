import pandas as pd
import os

directories = ['Experiment1', 'Experiment2', 'Experiment3', 'Experiment4']

if __name__ == "__main__":
    for d in directories:
      for filename in os.listdir(os.path.join('processed_experiments',d)):
        if os.path.isfile(os.path.join('processed_experiments',d,filename)) and filename.endswith('.csv'):
          df = pd.read_csv(os.path.join('processed_experiments',d,filename))
          df['overfitting'] = abs(df['te_mean']-df['tr_mean'])
          df = df.round(decimals=4)
          df.to_csv(os.path.join('processed_experiments',d,filename), index=False)