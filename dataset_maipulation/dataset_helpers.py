import pandas as pd
from typing import List

def transform_alpha_list_into_numeric(df: pd.DataFrame, name_list: List[str]):
  """Method that transforms a list of alphanumeric columns into numeric ones
  This is a wrapper for the transform_alpha_into_numeric method

  Arguments:
      df {pd.DataFrame} -- Dataframe to be altered
      name_list {List[str]} -- List of strings containing the columns to be 
      transformed.

  Returns:
      tuple -- Tuple including the mutated df and the list provided as argument.
      Note: That list can be useful in order to obtain the inputs and labels
      with extract_inputs_and_labels using it as a droplist
  """
  for cname in name_list:
    df = transform_alpha_into_numeric(df,cname)
  return (df,name_list)

def transform_alpha_into_numeric(df: pd.DataFrame, cname: str):
  """Method that transforms alphanumeric columns into numeric ones. this is done
  mainly to categorize the labels because sometimes they are alphanumeric ones
  instead of numeric ones

  Arguments:
      df {pd.DataFrame} -- Dataframe to be transformed
      cname {str} -- Name of the column to be transformed

  Returns:
      [type] -- Transformed dataframe
  """
  df[cname] = pd.Categorical(df[cname])
  df['new_{}'.format(cname)] = df[cname].cat.codes
  return df

def extract_inputs_and_labels(df:pd.DataFrame,
      labels_name:str, drop_list:List[str] = []):
  """Method that extracts the inputs and labels in order to use them in a neural
  network of this repository.

  Arguments:
      df {pd.DataFrame} -- Dataframe to be transformed into numpy arrays
      labels_name {str} -- The column name that contains the labels

  Keyword Arguments:
      drop_list {List[str]} -- List of columns that needs to be dropped prior to
      transforming the dataframe to numpy arrays (default: {[]})

  Returns:
      tuple -- A tuple containing inputs and labels in that order
  """
  # TODO: Add in a comment that this operation does not support multillabelling
  df_copy = df.copy()
  df_copy = df_copy.drop(columns=drop_list)
  labels = df_copy[labels_name].to_numpy()
  df_copy = df_copy.drop(columns=[labels_name])
  inputs = df_copy.to_numpy()
  return (inputs, labels)
