import pandas as pd
from typing import List

def transform_alpha_list_into_numeric(df: pd.DataFrame, name_list: List[str]):
  for cname in name_list:
    df = transform_alpha_into_numeric(df,cname)
  return (df,name_list)

def transform_alpha_into_numeric(df: pd.DataFrame, cname: str):
  df[cname] = pd.Categorical(df[cname])
  df['new_{}'.format(cname)] = df[cname].cat.codes
  return df

def extract_inputs_and_labels(df:pd.DataFrame,
      labels_name:str, drop_list:List[str] = []):
  # TODO: Add in a comment that this operation does not support multillabelling
  df_copy = df.copy()
  df_copy = df_copy.drop(columns=drop_list)
  labels = df_copy[labels_name].to_numpy()
  df_copy = df_copy.drop(columns=[labels_name])
  inputs = df_copy.to_numpy()
  return (inputs, labels)
