import ollama
import pandas as pd
import os
from nlp_synt_data import *
from data.prompts import *
from data.texts import *
from sc import *

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

  SEED = "seed-"
  SEED = ""

  # Generate synthetic data and run models
  prompts = PromptGenerator.generate(
    PROMPTS_JOB_V0, 
    # PROMPTS_JOB_V0_KEYS,
    [['zsl',],],
    )

  # data = DataGenerator.generate(TEXT_v0, SUBS_JOBS_V0)
  if SEED != "":
    data = DataGenerator.generate(
      TEXT_SEED_v0(), 
      SUBS_JOBS_V0)
  else:
    data = sc_make_test_data()

  # model = 'llama3:instruct'
  n_pass = 1
  model = 'gemma2'
  model = 'mistral'
  model = 'qwen2'
  model = 'llama3'
  model = 'phi3'
  for model in [
    # 'qwen2',
    # 'llama3.1',
    'llama3:instruct',
    'mistral',
    'gemma2',
    'phi3:medium',
    ]:
    print(model)
    model_f = lambda messages: ollama.chat(model=model, messages=messages)['message']['content']
    ResponseGenerator.generate(f"results/{model.split(':')[0]}_{SEED}split-long-v0.csv", 
                               data, 
                               prompts,
                               lambda prompt, text: PROMPTS_JOB_V0_F(prompt, text, model_f), 
                               n_pass=n_pass,
                               )
