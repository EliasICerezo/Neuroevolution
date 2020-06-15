import pandas as pd 
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

paths = ['Experiment1/experiment1.csv','Experiment2/experiment2.csv','Experiment3/experiment3.csv','Experiment4/experiment4.csv']
if __name__ == "__main__":
  for p in paths:
    df = pd.read_csv(p)
    isga = df['neural_network'] == 'GeneticNN'
    isbasic = df['neural_network'] == 'BasicNN'
    israndom = df['neural_network'] == 'RandomNN'
    ises = df['neural_network'] == 'StrategyNN'
    issa = df['neural_network'] == 'AnnealedNN'
    if any(isga) and any(isbasic) and any(israndom) and any(ises) and any(issa):
      ga = df[isga]
      basic = df[isbasic]
      randomnn = df[israndom]
      es = df[ises]
      sa = df[issa]
      fvalue, pvalue =stats.f_oneway(basic['training_loss'],randomnn['training_loss'],sa['training_loss'],es['training_loss'],ga['training_loss'])
      anova_training = "{} training | F: {} | P: {} | \n\n".format(p,fvalue,pvalue)
      print(anova_training)
      ga = ga.reset_index()
      es = es.reset_index()
      sa = sa.reset_index()
      basic = basic.reset_index()
      randomnn = randomnn.reset_index()

      ndf = pd.DataFrame()
      ndf['ga_training'] = ga['training_loss']
      ndf['es_training'] = es['training_loss']
      ndf['sa_training'] = sa['training_loss']
      ndf['basic_training'] = basic['training_loss']
      ndf['randomnn_training'] = randomnn['training_loss']
      d_melt = pd.melt(ndf.reset_index(), id_vars=['index'], value_vars=['ga_training','es_training','sa_training','basic_training','randomnn_training'])
      d_melt.columns = ['index', 'method', 'value']
      m_comp_training = pairwise_tukeyhsd(endog=d_melt['value'], groups=d_melt['method'], alpha=0.05)
      trdf = pd.DataFrame(data=m_comp_training._results_table.data[1:], columns=m_comp_training._results_table.data[0])
      trdf.to_csv('anovatrainingtable{}.csv'.format(p[-5]), index=False)
      


      ndf = pd.DataFrame()
      ndf['ga_testing'] = ga['testing_loss']
      ndf['es_testing'] = es['testing_loss']
      ndf['sa_testing'] = sa['testing_loss']
      ndf['basic_testing'] = basic['testing_loss']
      ndf['randomnn_testing'] = randomnn['testing_loss']
      ndf.replace([np.inf, -np.inf], np.nan)
      ndf = ndf[~ndf.isin([np.nan, np.inf, -np.inf]).any(1)]

      fvalue, pvalue =stats.f_oneway(ndf['ga_testing'],ndf['es_testing'],ndf['sa_testing'],ndf['basic_testing'],ndf['randomnn_testing'])
      anova_testing = "{} training | F: {} | P: {} | \n\n".format(p,fvalue,pvalue)
      
      d_melt = pd.melt(ndf.reset_index(), id_vars=['index'], value_vars=['ga_testing','es_testing','sa_testing','basic_testing','randomnn_testing'])
      d_melt.columns = ['index', 'method', 'value']
      m_comp_testing = pairwise_tukeyhsd(endog=d_melt['value'], groups=d_melt['method'], alpha=0.05)
      tedf = pd.DataFrame(data=m_comp_testing._results_table.data[1:], columns=m_comp_testing._results_table.data[0])
      tedf.to_csv('anovatestingtable{}.csv'.format(p[-5]), index=False)
      # breakpoint()
      file = open('anova{}.txt'.format(p[-5]), "w")
      file.write(anova_training)
      file.write(str(m_comp_training))
      file.write("\n\n")
      file.write(anova_testing)
      file.write(str(m_comp_testing))
      file.close()
