import ollama
import pandas as pd
import os
from nlp_synt_data import *
from data.prompts import *
from data.texts import *

# ollama run llama3:instruct
# ollama run mistral:instruct
# ollama run gemma:instruct
# ollama run phi3:instruct

models = [
  'llama3:instruct',
  'mistral:instruct',
  'gemma:instruct',
  'phi3:instruct'
  ]

# !wget https://github.com/fatemehmohammadi1995/DiscriminationDetection_UNIMI/raw/main/prompts.xlsx
# !wget https://github.com/samdsk/jobsearchapi/raw/dev/db/dbdump_testdb.jobs_v5.json


if __name__ == '__main__':

  if False:
    df = pd.read_json('data/dbdump_testdb.jobs_v5.json')
    titles = df.apply(lambda x: x['title'].lower(), axis=1).unique()
    # for title in titles:
    #   print(title)
    for d in df.iloc[34:45]['description']:
      print('##################')
      print(d)
    exit()

  # Generate synthetic data and run models
  # prompts = PromptGenerator.generate(PROMPTS_JOB_CIEQT_V1,[
  #                                     ['c','i','e','q'],
  #                                     ['c','i','e','t'],
  #                                     ['c','i','e','q','t'],
  #                                     ['c','e','i','q'],
  #                                     ])
  # prompts = PromptGenerator.generate(PROMPTS_COT_V0, PROMPTS_COT_V0_KEYS)
  prompts = PromptGenerator.generate(PROMPTS_JOB_SPLIT_V0, PROMPTS_JOB_SPLIT_V0_KEYS)
  
  # jobs = pd.read_csv('data/synt/jobs.csv').values.tolist()
  # data = DataGenerator.generate([(t,'none') for t in texts], {'JOB': [j for j in jobs]})

  texts = pd.read_csv('data/job_description_seed_dataset_improved_context.csv')#['text'].values.tolist()
  # data = DataGenerator.generate([(t,'none') for t in texts], {})
  # data = DataGenerator.generate([(t[1]['text'],t[1]['inclusive phrasing']) for t in texts.iterrows()], {})
  data = DataGenerator.generate(TEXT_JOBS_SONNET35, {
    # "JOB": JOBS_WITH_GENDER}
    "JOB": pd.read_csv('data/synt/jobs.csv').values.tolist()}
    )
  # model = 'llama3:instruct'
  model = 'gemma2'
  model = 'mistral'
  model = 'qwen2'
  for model in ['mistral','qwen2','gemma2']:
    ResponseGenerator.generate(f"results/{model}_split.csv", data, prompts,
                               lambda prompt, text: PROMPTS_JOB_SPLIT_V0_F(prompt, text, model), n_pass=10)
